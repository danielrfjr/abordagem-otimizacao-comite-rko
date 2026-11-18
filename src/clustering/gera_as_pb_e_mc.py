"""
gerar_comites.py
================
Gera 30 comitês independentes (10 partições-base cada) para cada combinação
(base × estratégia) e salva tudo em um único all_results.pkl, com salvamento
atômico incremental e retomada.


"""

import os
import sys
import json
import time
import math
import pickle
import hashlib
import argparse
import traceback
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler

from load_datasets import datasets, n_clusters, ground_truth
from particoes_base_e_matriz_de_coassociacao import (
    ESTRATEGIAS,
    PARAMETROS_ALGORITMOS,
    ComiteIncompletoError,
    gerar_comite_e_matrizes,
    calcular_matrizes,
    versoes_ambiente,
)

# ============================================================
# CONFIGURAÇÕES  (única seed manual: MASTER_SEED)
# ============================================================

MASTER_SEED       = 42
N_COMITES         = 30
N_PARTICOES       = 10
PADRONIZAR        = True
MAX_TENTATIVAS    = 2        # por partição (mesma seed)
N_BASES_ESPERADO  = 20

_RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PKL_PATH_PADRAO = os.path.join(_RAIZ, "results", "all_results.pkl")


# ============================================================
# Configuração do experimento (usada na checagem de compatibilidade)
# ============================================================

def _norm(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _sha256(X) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(X, dtype=float)).tobytes()).hexdigest()


def construir_config() -> dict:
    return {
        "master_seed": MASTER_SEED,
        "n_comites": N_COMITES,
        "n_particoes": N_PARTICOES,
        "padronizar": PADRONIZAR,
        "regra_k": "uniforme em {2, ..., floor(sqrt(n))}",
        "esquema_seeds": ("SeedSequence(entropy=MASTER_SEED, spawn_key=(crc32(base), id_estrategia, "
                          "comite, particao, proposito)); proposito 0=sorteio de k, 1=algoritmo"),
        "estrategias": {est: [a["nome"] for a in algs] for est, algs in ESTRATEGIAS.items()},
        "algoritmos_estocasticos": sorted({a["nome"] for algs in ESTRATEGIAS.values()
                                           for a in algs if a["estocastico"]}),
        "parametros_algoritmos": PARAMETROS_ALGORITMOS,
        "bases": {nome: {"shape": list(np.asarray(X).shape), "sha256": _sha256(X)}
                  for nome, X in datasets.items()},
    }


# ============================================================
# Persistência atômica e estado
# ============================================================

def salvar_atomico(dados: dict, path: str, tentativas: int = 20, espera: float = 0.5):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dados, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    ultimo_erro = None
    for i in range(tentativas):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:       # arquivo momentaneamente travado no Windows
            ultimo_erro = e
            time.sleep(espera * (1 + i * 0.5))
    raise PermissionError(
        f"Não foi possível substituir {path} após {tentativas} tentativas. "
        "Feche programas que possam estar usando o arquivo (notebook, OneDrive, antivírus). "
        f"O estado mais recente está em {tmp}."
    ) from ultimo_erro


def _estado_novo(config: dict) -> dict:
    return {
        "metadata": {
            "config": config,
            "ambiente": versoes_ambiente(),
            "criado_em": datetime.now().isoformat(timespec="seconds"),
        },
        "datasets": datasets,
        "n_clusters": n_clusters,
        "ground_truth": ground_truth,
        # all_results[base][estrategia]["committees"][comite_id] -> dict do comitê
        "all_results": {d: {e: {"committees": {}} for e in ESTRATEGIAS} for d in datasets},
        "falhas": [],
    }


def _diferencas(cfg_antiga, cfg_nova) -> list:
    if not isinstance(cfg_antiga, dict):
        return ["(arquivo legado ou sem metadados de configuração)"]
    a, b = _norm(cfg_antiga), _norm(cfg_nova)
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def carregar_ou_iniciar(path: str, config: dict, novo: bool) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            dados = pickle.load(f)
        meta = dados.get("metadata") if isinstance(dados, dict) else None
        cfg_antiga = meta.get("config") if isinstance(meta, dict) else None
        dif = _diferencas(cfg_antiga, config)

        if not dif:
            # compatível: remove qualquer comitê marcado incompleto
            for d in dados["all_results"]:
                for e in dados["all_results"][d]:
                    comites = dados["all_results"][d][e]["committees"]
                    for c in [c for c, v in comites.items()
                              if v.get("status") != "concluido" or len(v.get("partitions", [])) != N_PARTICOES]:
                        print(f"[AVISO] comitê incompleto removido: {d}|{e}|{c}")
                        del comites[c]
            dados["datasets"], dados["n_clusters"], dados["ground_truth"] = datasets, n_clusters, ground_truth
            dados.setdefault("falhas", [])
            return dados

        if not novo:
            print("\n" + "=" * 70)
            print(f"O arquivo existente ({path}) é INCOMPATÍVEL com a configuração atual.")
            print("Campos diferentes:", ", ".join(dif))
            print("Nada foi alterado. Para iniciar um NOVO experimento (o arquivo atual será")
            print("renomeado para backup, não apagado), execute com --novo-experimento.")
            print("=" * 70)
            sys.exit(1)

        backup = path.replace(".pkl", f".backup_{datetime.now():%Y%m%d_%H%M%S}.pkl")
        os.replace(path, backup)
        print(f"[INFO] arquivo anterior preservado em: {backup}")

    return _estado_novo(config)


# ============================================================
# Validações
# ============================================================

def validar_resultados(dados: dict, completo: bool = True) -> list:
    erros = []
    cfg = dados["metadata"]["config"]
    ar = dados["all_results"]
    P, C = cfg["n_particoes"], cfg["n_comites"]

    if completo and len(ar) != N_BASES_ESPERADO:
        erros.append(f"{len(ar)} bases (esperado {N_BASES_ESPERADO})")

    tot_c = tot_p = 0
    for d, por_est in ar.items():
        n = np.asarray(dados["datasets"][d]).shape[0]
        k_max = math.isqrt(n)
        if set(por_est) != set(ESTRATEGIAS):
            erros.append(f"{d}: estratégias {sorted(por_est)}")
        for e, bloco in por_est.items():
            comites = bloco["committees"]
            if completo and sorted(comites) != list(range(1, C + 1)):
                erros.append(f"{d}|{e}: comitês presentes {len(comites)}/{C}")
            for c, com in comites.items():
                tag = f"{d}|{e}|{c}"
                if com.get("status") != "concluido":
                    erros.append(f"{tag}: status={com.get('status')}")
                    continue
                parts, infos = com["partitions"], com["partition_info"]
                if len(parts) != P or len(infos) != P:
                    erros.append(f"{tag}: {len(parts)} partições / {len(infos)} infos (esperado {P})")
                    continue
                tot_c += 1
                tot_p += len(parts)
                for lab, inf in zip(parts, infos):
                    if np.asarray(lab).shape != (n,):
                        erros.append(f"{tag}: shape de labels inválido")
                    if not (2 <= inf["k_sorteado"] <= k_max):
                        erros.append(f"{tag}: k={inf['k_sorteado']} fora de [2,{k_max}]")
                    if inf["status"] != "ok":
                        erros.append(f"{tag}: partição com status {inf['status']}")
                co, dco = com["coassociation"], com["dissociation"]
                if co.shape != (n, n) or dco.shape != (n, n):
                    erros.append(f"{tag}: dimensões CO/DCO {co.shape}/{dco.shape}")
                    continue
                if not (np.allclose(co, co.T) and np.allclose(dco, dco.T)):
                    erros.append(f"{tag}: matriz assimétrica")
                if not np.allclose(np.diag(dco), 0):
                    erros.append(f"{tag}: diagonal da DCO != 0")
                if not np.allclose(np.diag(co), 1):
                    erros.append(f"{tag}: diagonal da CO != 1")
                if co.min() < 0 or co.max() > 1 or dco.min() < 0 or dco.max() > 1:
                    erros.append(f"{tag}: valores fora de [0,1]")
                if not np.allclose(co + dco, 1):
                    erros.append(f"{tag}: CO + DCO != 1")
                if not np.allclose(calcular_matrizes([np.asarray(l) for l in parts])[0], co):
                    erros.append(f"{tag}: CO armazenada != CO recalculada dos rótulos")

    if completo:
        if tot_c != N_BASES_ESPERADO * len(ESTRATEGIAS) * C:
            erros.append(f"total de comitês {tot_c} (esperado {N_BASES_ESPERADO * len(ESTRATEGIAS) * C})")
        if tot_p != N_BASES_ESPERADO * len(ESTRATEGIAS) * C * P:
            erros.append(f"total de partições {tot_p}")
    print(f"[VALIDAÇÃO] comitês concluídos={tot_c} | partições={tot_p} | erros={len(erros)}")
    return erros


# ============================================================
# Teste de reprodutibilidade (em memória; não grava o pkl)
# ============================================================

def teste_reprodutibilidade(n_comites: int = 3) -> bool:
    nome = min(datasets, key=lambda d: np.asarray(datasets[d]).shape[0])
    X = StandardScaler().fit_transform(np.asarray(datasets[nome], dtype=float))
    print(f"\n[TESTE] base='{nome}' n={X.shape[0]} | MASTER_SEED={MASTER_SEED} | {n_comites} comitês/estratégia")
    ok = True

    def gerar(est, c):
        return gerar_comite_e_matrizes(X, nome, est, c, MASTER_SEED, N_PARTICOES,
                                       padronizar=False, max_tentativas=MAX_TENTATIVAS)

    for est in ESTRATEGIAS:
        ids = list(range(1, n_comites + 1))
        run1 = {c: gerar(est, c) for c in ids}
        run2 = {c: gerar(est, c) for c in reversed(ids)}   # ordem invertida: testa independência de ordem

        # reprodutibilidade
        for c in ids:
            for p, (a, b) in enumerate(zip(run1[c]["partitions"], run2[c]["partitions"])):
                if not np.array_equal(a, b):
                    alg = run1[c]["partition_info"][p]["algoritmo"]
                    print(f"  [FALHA] {est} comitê {c} partição {p} ({alg}) difere entre as execuções")
                    ok = False
            ka = [i["k_sorteado"] for i in run1[c]["partition_info"]]
            kb = [i["k_sorteado"] for i in run2[c]["partition_info"]]
            if ka != kb:
                print(f"  [FALHA] {est} comitê {c}: ks diferentes")
                ok = False

        # diversidade entre comitês
        sigs = {tuple(l.tobytes() for l in run1[c]["partitions"]) for c in ids}
        ks = {tuple(i["k_sorteado"] for i in run1[c]["partition_info"]) for c in ids}
        print(f"  {est:16s} comitês distintos: {len(sigs)}/{n_comites} | sequências de k distintas: {len(ks)}/{n_comites}")
        if n_comites > 1 and len(sigs) < 2:
            print(f"  [FALHA] {est}: todos os comitês idênticos")
            ok = False

    print("[TESTE] " + ("OK: reprodutível e comitês diversos." if ok else "FALHOU."))
    if not ok:
        print("  Se a falha for em clara/clarans, o fastkmedoids pode usar RNG próprio sem argumento "
              "'seed' (ver 'clara_aceita_seed' nos metadados).")
    return ok


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=PKL_PATH_PADRAO)
    ap.add_argument("--novo-experimento", action="store_true")
    ap.add_argument("--teste", action="store_true")
    args = ap.parse_args()

    if args.teste:
        sys.exit(0 if teste_reprodutibilidade() else 1)

    if len(datasets) != N_BASES_ESPERADO:
        print(f"[AVISO] {len(datasets)} bases carregadas (esperado {N_BASES_ESPERADO}).")
    crcs = {}
    from particoes_base_e_matriz_de_coassociacao import id_estavel
    for nome in datasets:
        if id_estavel(nome) in crcs:
            raise RuntimeError(f"Colisão de id estável entre '{nome}' e '{crcs[id_estavel(nome)]}'.")
        crcs[id_estavel(nome)] = nome

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    config = construir_config()
    dados = carregar_ou_iniciar(args.output, config, args.novo_experimento)
    salvar_atomico(dados, args.output)

    total = len(datasets) * len(ESTRATEGIAS) * N_COMITES
    feitos = sum(len(dados["all_results"][d][e]["committees"]) for d in datasets for e in ESTRATEGIAS)
    print(f"Comitês concluídos: {feitos}/{total}")

    t_ini = time.time()
    novos = 0
    for nome, X in datasets.items():
        X_proc = StandardScaler().fit_transform(np.asarray(X, dtype=float)) if PADRONIZAR \
            else np.asarray(X, dtype=float)
        for est in ESTRATEGIAS:
            comites = dados["all_results"][nome][est]["committees"]
            for c in range(1, N_COMITES + 1):
                if c in comites:
                    continue
                t0 = time.time()
                try:
                    com = gerar_comite_e_matrizes(
                        X_proc, nome, est, c, MASTER_SEED, N_PARTICOES,
                        padronizar=False, max_tentativas=MAX_TENTATIVAS,
                    )
                except ComiteIncompletoError as e:
                    print(f"[ERRO] {e}", flush=True)
                    dados["falhas"] = [f for f in dados["falhas"]
                                       if not (f["dataset"] == nome and f["estrategia"] == est and f["comite"] == c)]
                    dados["falhas"].append({**e.info, "quando": datetime.now().isoformat(timespec="seconds")})
                    salvar_atomico(dados, args.output)
                    continue
                except Exception:
                    print(f"[ERRO INESPERADO] {nome}|{est}|{c}\n{traceback.format_exc()}", flush=True)
                    continue

                comites[c] = com
                dados["falhas"] = [f for f in dados["falhas"]
                                   if not (f["dataset"] == nome and f["estrategia"] == est and f["comite"] == c)]
                salvar_atomico(dados, args.output)
                novos += 1
                print(f"[{nome}|{est}|comitê {c:02d}] ok em {time.time()-t0:.1f}s "
                      f"| total {feitos + novos}/{total}", flush=True)

    # ------------------ finalização ------------------
    pendentes = [(d, e, c) for d in datasets for e in ESTRATEGIAS
                 for c in range(1, N_COMITES + 1)
                 if c not in dados["all_results"][d][e]["committees"]]
    print(f"\nNovos nesta execução: {novos} | tempo: {(time.time()-t_ini)/60:.1f} min | pendentes: {len(pendentes)}")
    if dados["falhas"]:
        print(f"{len(dados['falhas'])} falha(s) registradas em dados['falhas']. Rode novamente para retentar.")

    # recarrega do disco para provar que o arquivo é legível
    with open(args.output, "rb") as f:
        recarregado = pickle.load(f)
    erros = validar_resultados(recarregado, completo=not pendentes)
    for e in erros[:30]:
        print("  -", e)
    if pendentes or erros:
        sys.exit(2)
    print(f"Experimento completo e validado: {args.output}")


if __name__ == "__main__":
    main()
