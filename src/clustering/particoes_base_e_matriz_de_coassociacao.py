"""
particoes_base_e_matriz_de_coassociacao.py
==========================================
Geração de UM comitê de agrupamentos (10 partições-base) para uma combinação
(base, estratégia, comitê), usando pacotes R via rpy2, com derivação
determinística de seeds a partir de uma MASTER_SEED.

Estratégia 1 — Não hierárquicos: kmeans (HW, Lloyd, MacQueen), PAM, CLARA, CLARANS
Estratégia 2 — Hierárquicos: single, average, complete, ward.D2, median,
               centroid, mcquitty (stats::hclust) e DIANA (cluster::diana)
Estratégia 3 — Mistura: lista combinada das duas anteriores.

Dependências Python: numpy scikit-learn rpy2
Dependências R:      install.packages(c("cluster", "fastkmedoids"))

Notas
-----
- rpy2 não é thread-safe: não usar com paralelismo.
- Labels do R são 1-based; convertidos para 0-based.
- DCO = 1 - CO; diagonal da DCO é 0.
"""

import os
import time
import math
import zlib
import platform
from itertools import cycle, islice

import numpy as np
from sklearn.preprocessing import StandardScaler

# --- configuração do R (Windows) ---
os.environ["R_HOME"]   = r"C:\Program Files\R\R-4.6.0"
os.environ["R_LIBS"]   = r"C:\Program Files\R\R-4.6.0\library"
os.environ["PATH"]     = r"C:\Program Files\R\R-4.6.0\bin\x64;" + os.environ["PATH"]
os.environ["LANGUAGE"] = "en"
os.environ["LC_ALL"]   = "C"

# ---------------------------------------------------------------------------
# Inicialização do rpy2 e pacotes R
# ---------------------------------------------------------------------------
try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    import rpy2.rinterface_lib.callbacks as cb

    cb.consolewrite_warnerror = lambda s: None
    cb.consolewrite_print     = lambda s: None

    _r_base    = importr("base")
    _r_stats   = importr("stats")
    _r_cluster = importr("cluster")

    try:
        _r_fastkmedoids = importr("fastkmedoids")
    except Exception:
        raise ImportError(
            "Pacote R 'fastkmedoids' não encontrado.\n"
            "Instale com: install.packages('fastkmedoids')"
        )

except ImportError as e:
    raise ImportError(
        "rpy2 ou pacotes R não encontrados.\n"
        "Instale rpy2: pip install rpy2\n"
        "Instale pacotes R: install.packages(c('cluster','fastkmedoids'))\n"
        f"Erro original: {e}"
    )

_r_set_seed = ro.r["set.seed"]


# ---------------------------------------------------------------------------
# Seeds: derivação determinística a partir da MASTER_SEED
# ---------------------------------------------------------------------------

ESTRATEGIA_IDS = {"nao_hierarquica": 1, "hierarquica": 2, "mistura": 3}
PROPOSITO_K    = 0   # fluxo do sorteio de k
PROPOSITO_ALGO = 1   # fluxo do algoritmo de agrupamento (se estocástico)


def id_estavel(nome: str) -> int:
    """Identificador inteiro estável (independe do processo) para um nome."""
    return zlib.crc32(nome.encode("utf-8"))


def derivar_seed(master_seed: int, dataset: str, estrategia: str,
                 comite_id: int, particao_idx: int, proposito: int) -> int:
    """
    Seed de 32 bits derivada de (MASTER_SEED, base, estratégia, comitê,
    partição, propósito) via SeedSequence. Não depende da ordem de execução.
    """
    ss = np.random.SeedSequence(
        entropy=int(master_seed),
        spawn_key=(id_estavel(dataset), ESTRATEGIA_IDS[estrategia],
                   int(comite_id), int(particao_idx), int(proposito)),
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _definir_seed_r(seed: int):
    _r_set_seed(int(seed))


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _relabel(labels: np.ndarray) -> np.ndarray:
    """Renumera rótulos para 0, 1, 2, ... sem lacunas."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(int)


def _to_r_matrix(X: np.ndarray):
    return ro.r.matrix(
        ro.FloatVector(X.flatten()),
        nrow=int(X.shape[0]),
        ncol=int(X.shape[1]),
        byrow=True,
    )


def _to_r_dist(X: np.ndarray):
    return _r_stats.dist(_to_r_matrix(X), method="euclidean")


def _labels_from_r(r_result, field: str) -> np.ndarray:
    return _relabel(np.array(r_result.rx2(field), dtype=int) - 1)


def _r_aceita_seed(r_func) -> bool:
    """True se a função R possui argumento 'seed'."""
    try:
        nomes = ro.r("function(f) names(formals(f))")(r_func)
        return "seed" in [str(x) for x in nomes]
    except Exception:
        return False


_CLARA_ACEITA_SEED   = _r_aceita_seed(_r_fastkmedoids.fastclara)
_CLARANS_ACEITA_SEED = _r_aceita_seed(_r_fastkmedoids.fastclarans)


# ---------------------------------------------------------------------------
# Estratégia 1 — Não hierárquicos via R
# Todas as funções aceitam seed=None (compatível com chamadas func(X, k)).
# Nos K-Means, a aleatoriedade vem do RNG global do R (set.seed no runner).
# ---------------------------------------------------------------------------

def _kmeans_hartigan_wong(X, k, seed=None):
    result = _r_stats.kmeans(_to_r_matrix(X), centers=k,
                             algorithm="Hartigan-Wong", nstart=10)
    return _labels_from_r(result, "cluster")


def _kmeans_lloyd(X, k, seed=None):
    result = _r_stats.kmeans(_to_r_matrix(X), centers=k,
                             algorithm="Lloyd", nstart=10)
    return _labels_from_r(result, "cluster")


def _kmeans_macqueen(X, k, seed=None):
    result = _r_stats.kmeans(_to_r_matrix(X), centers=k,
                             algorithm="MacQueen", nstart=10)
    return _labels_from_r(result, "cluster")


def _pam(X, k, seed=None):
    n = int(X.shape[0])
    result = _r_fastkmedoids.pam(_to_r_dist(X), n=n, k=k)
    labels = np.array(result.slots["assignment"], dtype=int) - 1
    return _relabel(labels)


def _clara(X, k, seed=None):
    n = int(X.shape[0])
    kwargs = {"seed": int(seed)} if (seed is not None and _CLARA_ACEITA_SEED) else {}
    result = _r_fastkmedoids.fastclara(_to_r_dist(X), n=n, k=k, **kwargs)
    labels = np.array(result.slots["assignment"], dtype=int) - 1
    return _relabel(labels)


def _clarans(X, k, seed=None):
    n = int(X.shape[0])
    kwargs = {"seed": int(seed)} if (seed is not None and _CLARANS_ACEITA_SEED) else {}
    result = _r_fastkmedoids.fastclarans(_to_r_dist(X), n=n, k=k, **kwargs)
    labels = np.array(result.slots["assignment"], dtype=int) - 1
    return _relabel(labels)


# ---------------------------------------------------------------------------
# Estratégia 2 — Hierárquicos via R (determinísticos)
# ---------------------------------------------------------------------------

def _hclust(metodo_r: str):
    def _func(X, k, seed=None):
        r_hclust = _r_stats.hclust(_to_r_dist(X), method=metodo_r)
        r_labels = _r_stats.cutree(r_hclust, k=k)
        return _relabel(np.array(r_labels, dtype=int) - 1)
    _func.__name__ = metodo_r
    return _func


def _diana(X, k, seed=None):
    result   = _r_cluster.diana(_to_r_matrix(X))
    r_labels = _r_stats.cutree(result, k=k)
    return _relabel(np.array(r_labels, dtype=int) - 1)


# ---------------------------------------------------------------------------
# Listas de algoritmos (mesma ordem da versão anterior)
# "estocastico": True => o runner define set.seed (R) e seed derivada.
# ---------------------------------------------------------------------------

NAO_HIERARQUICOS = [
    {"nome": "kmeans_hartigan_wong", "func": _kmeans_hartigan_wong, "estocastico": True},
    {"nome": "kmeans_lloyd",         "func": _kmeans_lloyd,         "estocastico": True},
    {"nome": "kmeans_macqueen",      "func": _kmeans_macqueen,      "estocastico": True},
    {"nome": "pam",                  "func": _pam,                  "estocastico": False},
    {"nome": "clara",                "func": _clara,                "estocastico": True},
    {"nome": "clarans",              "func": _clarans,              "estocastico": True},
]

HIERARQUICOS = [
    {"nome": "single",   "func": _hclust("single"),   "estocastico": False},
    {"nome": "average",  "func": _hclust("average"),  "estocastico": False},
    {"nome": "complete", "func": _hclust("complete"), "estocastico": False},
    {"nome": "ward",     "func": _hclust("ward.D2"),  "estocastico": False},
    {"nome": "median",   "func": _hclust("median"),   "estocastico": False},
    {"nome": "centroid", "func": _hclust("centroid"), "estocastico": False},
    {"nome": "mcquitty", "func": _hclust("mcquitty"), "estocastico": False},
    {"nome": "diana",    "func": _diana,              "estocastico": False},
]

MISTURA = NAO_HIERARQUICOS + HIERARQUICOS

ESTRATEGIAS = {
    "nao_hierarquica": NAO_HIERARQUICOS,
    "hierarquica":     HIERARQUICOS,
    "mistura":         MISTURA,
}

# Parâmetros relevantes (registrados nos metadados e usados na checagem de compatibilidade)
PARAMETROS_ALGORITMOS = {
    "kmeans_hartigan_wong": {"pacote": "stats::kmeans", "algorithm": "Hartigan-Wong", "nstart": 10},
    "kmeans_lloyd":         {"pacote": "stats::kmeans", "algorithm": "Lloyd", "nstart": 10},
    "kmeans_macqueen":      {"pacote": "stats::kmeans", "algorithm": "MacQueen", "nstart": 10},
    "pam":     {"pacote": "fastkmedoids::pam", "entrada": "dist euclidiana"},
    "clara":   {"pacote": "fastkmedoids::fastclara", "entrada": "dist euclidiana"},
    "clarans": {"pacote": "fastkmedoids::fastclarans", "entrada": "dist euclidiana"},
    "single":   {"pacote": "stats::hclust", "method": "single"},
    "average":  {"pacote": "stats::hclust", "method": "average"},
    "complete": {"pacote": "stats::hclust", "method": "complete"},
    "ward":     {"pacote": "stats::hclust", "method": "ward.D2"},
    "median":   {"pacote": "stats::hclust", "method": "median"},
    "centroid": {"pacote": "stats::hclust", "method": "centroid"},
    "mcquitty": {"pacote": "stats::hclust", "method": "mcquitty"},
    "diana":    {"pacote": "cluster::diana", "entrada": "matriz de dados"},
}

def _versao_pacote(nome: str) -> str:
    """Versão instalada de um pacote Python, sem depender de __version__."""
    try:
        from importlib.metadata import version
        return version(nome)
    except Exception:
        return "desconhecida"
    
def versoes_ambiente() -> dict:
    import sklearn
    v = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "rpy2": _versao_pacote("rpy2"),
        "clara_aceita_seed": _CLARA_ACEITA_SEED,
        "clarans_aceita_seed": _CLARANS_ACEITA_SEED,
    }
    try:
        v["R"] = str(ro.r("R.version.string")[0])
        for pkg in ("cluster", "fastkmedoids"):
            v[f"R::{pkg}"] = str(ro.r(f'as.character(packageVersion("{pkg}"))')[0])
    except Exception as e:
        v["R_erro"] = str(e)
    return v


# ---------------------------------------------------------------------------
# CO / DCO
# ---------------------------------------------------------------------------

def calcular_matrizes(particoes: list) -> tuple:
    """CO[i,j] = fração das partições em que i e j estão no mesmo grupo; DCO = 1 - CO."""
    P = len(particoes)
    n = len(particoes[0])
    coassoc = np.zeros((n, n), dtype=float)
    for lab in particoes:
        coassoc += (lab[:, None] == lab[None, :])
    coassoc /= P
    return coassoc, 1.0 - coassoc


# ---------------------------------------------------------------------------
# Geração de um comitê
# ---------------------------------------------------------------------------

class ComiteIncompletoError(Exception):
    """Falha em uma partição; o comitê NÃO deve ser considerado concluído."""
    def __init__(self, info: dict):
        self.info = info
        super().__init__(
            f"[{info['dataset']}|{info['estrategia']}|comite {info['comite']}|"
            f"particao {info['particao']}|{info['algoritmo']}|k={info['k_sorteado']}] "
            f"{info['erro']}"
        )


def gerar_comite_e_matrizes(
    X: np.ndarray,
    dataset: str,
    estrategia: str,
    comite_id: int,
    master_seed: int = 42,
    n_particoes: int = 10,
    padronizar: bool = True,
    max_tentativas: int = 2,
    verbose: bool = False,
) -> dict:
    """
    Gera UM comitê (n_particoes partições-base) para (dataset, estrategia, comite_id).

    Pode levantar ComiteIncompletoError. Nunca substitui um algoritmo por outro.
    Com `padronizar=True` o Z-score é aplicado aqui; o script principal passa
    dados já padronizados com padronizar=False para não recalculá-lo.
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    X_proc = StandardScaler().fit_transform(X) if padronizar else X

    k_max = math.isqrt(n)
    if k_max < 2:
        raise ValueError(f"Base '{dataset}' (n={n}) pequena demais para k em [2, floor(sqrt(n))].")

    roteiro = list(islice(cycle(ESTRATEGIAS[estrategia]), n_particoes))
    particoes, info = [], []
    t_comite = time.perf_counter()

    for idx, item in enumerate(roteiro):
        # --- sorteio de k (fluxo próprio) ---
        seed_k = derivar_seed(master_seed, dataset, estrategia, comite_id, idx, PROPOSITO_K)
        k = int(np.random.default_rng(seed_k).integers(2, k_max + 1))

        # --- seed do algoritmo (somente se estocástico) ---
        seed_algo = None
        if item["estocastico"]:
            seed_algo = derivar_seed(master_seed, dataset, estrategia, comite_id,
                                     idx, PROPOSITO_ALGO) & 0x7FFFFFFF  # set.seed do R: int32 positivo

        log(f"  [{idx+1:02d}/{n_particoes}] {item['nome']:22s} k={k}")

        labels, erro, tentativas = None, None, 0
        for tentativas in range(1, max_tentativas + 1):
            try:
                if seed_algo is not None:
                    _definir_seed_r(seed_algo)       # aleatoriedade executada no R
                    np.random.seed(seed_algo)        # aleatoriedade executada no Python
                t0 = time.perf_counter()
                labels = np.asarray(item["func"](X_proc, k, seed=seed_algo), dtype=int)
                elapsed = time.perf_counter() - t0
                if labels.shape != (n,):
                    raise ValueError(f"labels com shape {labels.shape}, esperado ({n},)")
                erro = None
                break
            except Exception as e:
                erro = f"{type(e).__name__}: {e}"
                labels = None

        if erro is not None:
            raise ComiteIncompletoError({
                "dataset": dataset, "estrategia": estrategia, "comite": comite_id,
                "particao": idx, "algoritmo": item["nome"], "k_sorteado": k,
                "seed_k": seed_k, "seed_algoritmo": seed_algo,
                "tentativas": tentativas, "erro": erro,
            })

        particoes.append(labels)
        info.append({
            "dataset": dataset, "estrategia": estrategia, "comite": comite_id,
            "particao": idx, "algoritmo": item["nome"],
            "k_sorteado": k, "k_max": k_max, "k_real": int(np.unique(labels).shape[0]),
            "seed_k": seed_k, "seed_algoritmo": seed_algo,
            "tempo_s": round(elapsed, 4), "tentativas": tentativas,
            "status": "ok", "erro": None,
        })

    coassoc, dissoc = calcular_matrizes(particoes)
    return {
        "status": "concluido",
        "comite_id": comite_id,
        "partitions": particoes,
        "partition_info": info,
        "coassociation": coassoc,
        "dissociation": dissoc,
        "tempo_total_s": round(time.perf_counter() - t_comite, 4),
    }