import numpy as np
from sklearn.metrics import silhouette_score
from rko import RKO, RKOEnvAbstract


class ClusteringEnsemble(RKOEnvAbstract):
    """
    Ambiente do problema de Comitê de Agrupamentos para o RKO.

    O cromossomo tem k chaves aleatórias, uma por medoide.
    Cada chave ri é convertida no índice do medoide pela fórmula:
        λi = ceil(n * ri)   (indexado em 1, convertido para 0 ao acessar D)

    Parâmetros
    ----------
    dissimilarity_matrix : np.ndarray (n, n)
        Matriz de dissociação DCO pré-computada.
    k : int
        Número de clusters (fixo por instância).
    instance_name : str
        Nome da instância (usado nos logs do RKO).
    max_time : int
        Tempo máximo de execução em segundos.
    kappa : int
        Número de vizinhos mais próximos dentro do grupo considerados
        no procedimento iterativo da Fase 2.  κ=10.
    """

    def __init__(
        self,
        dissimilarity_matrix: np.ndarray,
        k: int,
        instance_name: str = "clustering",
        max_time: int = 120,
        kappa: int = 10,
    ):
        super().__init__()

        self.D             = dissimilarity_matrix
        self.n             = dissimilarity_matrix.shape[0]
        self.k             = k
        self.instance_name = instance_name
        self.max_time      = max_time
        self.kappa         = kappa

        # tamanho do cromossomo: uma key por medoide
        self.tam_solution  = k

        self.LS_type       = 'Best'
        self.dict_best     = {}
        self.save_q_learning_report = False

        # ------------------------------------------------------------------
        # Parâmetros das metaheurísticas
        # Q-Learning ativado automaticamente quando há mais de 1 valor por
        # parâmetro. Com 3 valores cada:
        #   BRKGA: 3^4 = 81 estados
        #   VNS:   3^2 =  9 estados
        #   ILS:   3^2 =  9 estados
        # ------------------------------------------------------------------

        self.BRKGA_parameters = {
            # tamanho da população 
            'p':    [50, 75, 100],
            # fração elite
            'pe':   [0.10, 0.20, 0.25],
            # probabilidade de herdar do elite 
            'rhoe': [0.60, 0.70, 0.80],
        }

        self.VNS_parameters = {
            'kMax':    [3, 5, 7],
            'betaMin': [0.05, 0.10, 0.20],
        }

        self.ILS_parameters = {
            'betaMin': [0.05, 0.10, 0.20],
            'betaMax': [0.20, 0.30, 0.40],
        }

    # ------------------------------------------------------------------
    # DECODER
    # λi = ceil(n * ri)  →  índice do medoide (1-based → convertido p/ 0-based)
    # Medoides duplicados são substituídos pelo objeto mais distante
    # dos medoides já selecionados.
    # ------------------------------------------------------------------
    def decoder(self, keys: np.ndarray):
        """
        Converte o cromossomo de k chaves nos índices dos medoides.

        Retorna
        -------
        labels   : np.ndarray (n,) — rótulo de cluster de cada objeto
        medoids  : list[int]       — índices dos medoides (0-based)
        """
        medoids = []
        for ri in keys:
            idx = int(np.ceil(self.n * ri)) - 1   # ceil(n*r) convertido p/ 0-based
            idx = max(0, min(idx, self.n - 1))     # garante bounds
            medoids.append(idx)

        # resolve duplicatas: substitui pelo objeto mais distante
        # dos medoides já selecionados
        medoids_unicos = []
        usados = set()

        for m in medoids:
            if m not in usados:
                medoids_unicos.append(m)
                usados.add(m)
            else:
                candidatos = [o for o in range(self.n) if o not in usados]
                if candidatos:
                    novo = max(
                        candidatos,
                        key=lambda o: min(self.D[o, j] for j in usados)
                    )
                    medoids_unicos.append(novo)
                    usados.add(novo)

        medoids = medoids_unicos

        # atribui cada objeto ao medoide mais próximo
        D_medoids = self.D[:, medoids]            # (n, k)
        labels    = np.argmin(D_medoids, axis=1)  # (n,)

        return labels, medoids

    # ------------------------------------------------------------------
    # COST  —  negativo da silhueta (RKO minimiza)
    # ------------------------------------------------------------------
    def cost(self, solution, final_solution: bool = False) -> float:
        """
        Avalia a partição pelo índice de silhueta médio.

        Retorna  −silhouette  (RKO minimiza → maximiza silhueta).

        Casos degenerados retornam 1.0 (pior penalidade possível),
        pois −silhueta ∈ [−1, 1] e silhueta=−1 é o pior caso real.
        """
        labels, medoids = solution
        k_real = len(np.unique(labels))

        # casos degenerados: silhueta indefinida → penalidade máxima
        if k_real < 2 or k_real >= self.n:
            return 1.0

        try:
            sil = silhouette_score(self.D, labels, metric='precomputed')
        except Exception:
            return 1.0

        if final_solution:
            print(f"\n{'='*50}")
            print(f"  Instância : {self.instance_name}")
            print(f"  k         : {k_real}")
            print(f"  Medoides  : {medoids}")
            print(f"  Silhueta  : {sil:.6f}")
            print(f"{'='*50}\n")

        return float(-sil)

    # ------------------------------------------------------------------
    # FASE 2 — Procedimento iterativo otimizado 
    #
    # 
    #
    # 1. Tabela de distâncias dist_obj_grupo (n × k) mantida
    #    incrementalmente — só atualiza a coluna do grupo que recebeu
    #    um novo objeto, não recalcula tudo do zero.
    #
    # 2. sum_knn (n × k) acumula a soma das κ menores distâncias de
    #    cada objeto a cada grupo. Após cada alocação, apenas a coluna
    #    do grupo afetado é recalculada usando operações numpy vetorizadas.
    #
    # 3. O par (objeto, grupo) ótimo é encontrado com np.unravel_index
    #    sobre a matriz dist_obj_grupo — sem loop Python sobre objetos.
    #
    # 
    # ------------------------------------------------------------------
    def procedimento_iterativo(self, medoids: list, verbose: bool = False) -> np.ndarray:
        k   = len(medoids)
        n   = self.n
        kap = self.kappa

        labels = np.full(n, -1, dtype=int)
        for r, m in enumerate(medoids):
            labels[m] = r

        grupos = [np.array([m], dtype=int) for m in medoids]

        nao_alocado = np.ones(n, dtype=bool)
        for m in medoids:
            nao_alocado[m] = False

        dist_obj_grupo = np.full((n, k), np.inf)
        for r, m in enumerate(medoids):
            dist_obj_grupo[:, r] = self.D[:, m]

        idx_nao_alocados = np.where(nao_alocado)[0]

        if verbose:
            print(f"\n{'='*55}")
            print(f"PROCEDIMENTO ITERATIVO — κ={kap}")
            print(f"{'='*55}")
            print(f"Medoides iniciais : {medoids}")
            print(f"Grupos iniciais   : {[list(g) for g in grupos]}")
            print(f"Não alocados      : {list(idx_nao_alocados)}")

        iteracao = 0
        while len(idx_nao_alocados) > 0:
            iteracao += 1

            sub      = dist_obj_grupo[idx_nao_alocados, :]
            pos_flat = np.argmin(sub)
            pos_xi, pos_r = np.unravel_index(pos_flat, sub.shape)
            xi       = idx_nao_alocados[pos_xi]
            r        = pos_r

            if verbose:
                print(f"\n--- Iteração {iteracao} ---")
                print(f"Distâncias objeto→grupo (não alocados):")
                for i, obj in enumerate(idx_nao_alocados):
                    dists_str = "  ".join(f"G{rr}={sub[i,rr]:.3f}" for rr in range(k))
                    marker = " ← MENOR" if obj == xi else ""
                    print(f"  obj {obj}: {dists_str}{marker}")
                print(f"Alocando objeto {xi} → grupo {r}")

            labels[xi] = r
            grupos[r]  = np.append(grupos[r], xi)
            nao_alocado[xi] = False

            grupo_r   = grupos[r]
            tam_grupo = len(grupo_r)
            k_efetivo = min(kap, tam_grupo)

            idx_nao_alocados = np.where(nao_alocado)[0]

            if len(idx_nao_alocados) == 0:
                break

            dists_ao_grupo = self.D[np.ix_(idx_nao_alocados, grupo_r)]

            if k_efetivo == tam_grupo:
                dist_obj_grupo[idx_nao_alocados, r] = dists_ao_grupo.mean(axis=1)
            else:
                idx_knn = np.argpartition(dists_ao_grupo, k_efetivo - 1, axis=1)[:, :k_efetivo]
                rows    = np.arange(len(idx_nao_alocados))[:, None]
                dist_obj_grupo[idx_nao_alocados, r] = dists_ao_grupo[rows, idx_knn].mean(axis=1)

            if verbose:
                print(f"Grupos agora      : {[list(g) for g in grupos]}")
                print(f"Não alocados      : {list(idx_nao_alocados)}")

        if verbose:
            print(f"\n{'='*55}")
            print(f"RESULTADO FINAL")
            print(f"{'='*55}")
            for r, grupo in enumerate(grupos):
                print(f"  Grupo {r}: {list(grupo)}")
            print(f"  Labels : {labels}")

        return np.array(labels, dtype=int)