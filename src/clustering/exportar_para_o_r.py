# %% [markdown]
# # Exportar partições-base do pkl para o R (v3)
#
# Para cada (dataset, estratégia, comitê) gera um CSV com a matriz PI
# (n objetos x B partições-base; rótulos inteiros 1-based), no formato esperado
# por CE_EAC / CE_WEAC / CE_LWEAC.
#
# Saída: results/partitions_for_R/<dataset>/<estrategia>/cNN_PI.csv
#        results/partitions_for_R/<dataset>_ground_truth.csv
#        results/partitions_for_R/_resumo_datasets.csv
#        results/partitions_for_R/_manifesto_particoes.csv
#        results/partitions_for_R/_metadata.json

# %%
import os
import json
import pickle
import numpy as np
import pandas as pd

# raiz do projeto = dois níveis acima de src/clustering/ (ajuste se o arquivo estiver em outro lugar)
try:
    _AQUI = os.path.dirname(os.path.abspath(__file__))
    RAIZ = os.path.abspath(os.path.join(_AQUI, '..', '..'))
except NameError:                       # rodando em notebook (sem __file__), a partir de notebooks/
    RAIZ = os.path.abspath('..')

PKL_PATH = os.path.join(RAIZ, 'results', 'all_results.pkl')
OUT_DIR  = os.path.join(RAIZ, 'results', 'partitions_for_R')
SOBRESCREVER = False                    # True = regrava arquivos já exportados
os.makedirs(OUT_DIR, exist_ok=True)

print(f'Carregando {PKL_PATH} (pode demorar e usar vários GB de RAM)...', flush=True)
with open(PKL_PATH, 'rb') as f:
    dados = pickle.load(f)

all_results  = dados['all_results']
n_clusters   = dados['n_clusters']
ground_truth = dados['ground_truth']
meta         = dados['metadata']['config']
N_COMITES    = meta['n_comites']
N_PARTICOES  = meta['n_particoes']

# libera a coassociação (não é exportada)
for d in all_results:
    for e in all_results[d]:
        for com in all_results[d][e]['committees'].values():
            com.pop('coassociation', None)

# exige comitês completos
faltando = [(d, e, c) for d in all_results for e in all_results[d]
            for c in range(1, N_COMITES + 1)
            if c not in all_results[d][e]['committees']]
if faltando:
    raise RuntimeError(f'{len(faltando)} comitês ausentes (ex.: {faltando[:3]}). Termine a geração antes.')

# %%
resumo, manifesto = [], []
exportados = pulados = 0

for dataset, estrategias in all_results.items():
    y_true = np.asarray(ground_truth[dataset])
    n = len(y_true)

    pd.DataFrame({'y_true': y_true}).to_csv(
        os.path.join(OUT_DIR, f'{dataset}_ground_truth.csv'), index=False)
    resumo.append({'dataset': dataset, 'k': n_clusters[dataset], 'n': n})

    for estrategia, bloco in estrategias.items():
        pasta = os.path.join(OUT_DIR, dataset, estrategia)
        os.makedirs(pasta, exist_ok=True)

        for c in range(1, N_COMITES + 1):
            com = bloco['committees'][c]
            partitions = com['partitions']
            if len(partitions) != N_PARTICOES:
                raise RuntimeError(f'{dataset}|{estrategia}|c{c}: {len(partitions)} partições')

            # manifesto (sempre, é leve)
            for inf in com['partition_info']:
                manifesto.append({
                    'dataset': dataset, 'estrategia': estrategia, 'comite': c,
                    'particao': inf['particao'] + 1, 'algoritmo': inf['algoritmo'],
                    'k_sorteado': inf['k_sorteado'], 'k_real': inf['k_real'],
                    'seed_k': inf['seed_k'], 'seed_algoritmo': inf['seed_algoritmo'],
                })

            arq = os.path.join(pasta, f'c{c:02d}_PI.csv')
            if os.path.exists(arq) and not SOBRESCREVER:
                pulados += 1
                continue

            PI = np.column_stack(partitions) + 1          # (n, B), 1-based
            assert PI.shape == (n, N_PARTICOES), (dataset, estrategia, c, PI.shape)

            tmp = arq + '.tmp'
            pd.DataFrame(PI, columns=[f'b{i+1}' for i in range(PI.shape[1])]).to_csv(tmp, index=False)
            os.replace(tmp, arq)
            exportados += 1

    print(f'{dataset} ok', flush=True)

pd.DataFrame(resumo).to_csv(os.path.join(OUT_DIR, '_resumo_datasets.csv'), index=False)
pd.DataFrame(manifesto).to_csv(os.path.join(OUT_DIR, '_manifesto_particoes.csv'), index=False)
with open(os.path.join(OUT_DIR, '_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'comites_master_seed': meta['master_seed'],
        'n_comites': N_COMITES,
        'n_particoes': N_PARTICOES,
        'estrategias': list(next(iter(all_results.values())).keys()),
        'formato_PI': 'n x B, rotulos inteiros 1-based, colunas b1..bB',
        'origem': 'all_results.pkl',
    }, f, indent=2, ensure_ascii=False)

print(f'\nExportados: {exportados} | pulados (já existiam): {pulados}')
print(f'Total esperado: {len(all_results) * 3 * N_COMITES} arquivos PI em {OUT_DIR}')