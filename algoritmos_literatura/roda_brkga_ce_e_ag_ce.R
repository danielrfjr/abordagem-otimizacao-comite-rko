################################################################################
### Literatura estocástica: BRKGA_CE e AG8
### 1 execução por (base, estratégia, comitê, método). 
################################################################################

library(cluster)
library(aricode)
library(mclust)
library(jsonlite)

# ------------------------------------------------------------------
# SEED MASTER (única seed manual deste script)
# ------------------------------------------------------------------
R_MASTER_SEED <- 42

# ------------------------------------------------------------------
# Caminhos
# O script deve ser executado de dentro de algoritmos_literatura/
# (o diretório de trabalho precisa conter os arquivos .R dos algoritmos).
# Estrutura do projeto:
#   projeto/
#     algoritmos_literatura/   <- diretório de trabalho
#     results/                 <- partitions_for_R/ (entrada) e CSVs/JSON (saída)
#     datasets/, src/, ...
# ------------------------------------------------------------------
SRC_DIR     <- "."                                     # arquivos .R dos algoritmos estão na pasta atual
RESULTS_DIR <- file.path("..", "results")              # results/ fica um nível acima
PART_DIR    <- file.path(RESULTS_DIR, "partitions_for_R")

OUT_CSV     <- file.path(RESULTS_DIR, "resultados_literatura_brkga_ce_ga_ce.csv")
LABELS_CSV  <- file.path(RESULTS_DIR, "labels_literatura_brkga_ce_ga_ce.csv")
ERROS_CSV   <- file.path(RESULTS_DIR, "erros_literatura_brkga_ce_ga_ce.csv")
CONFIG_JSON <- file.path(RESULTS_DIR, "config_literatura_brkga_ce_ga_ce.json")

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Hash determinístico de string (base R, sem dependências): estável entre
# execuções e máquinas. Mesma ideia do crc32 do Python, mas valores distintos.
hash_str <- function(s) {
  h <- 5381
  for (ch in utf8ToInt(s)) h <- (h * 33 + ch) %% 2147483647
  h
}
seed_lit <- function(dataset, estrategia, metodo, comite) {
  as.integer(hash_str(paste(R_MASTER_SEED, "lit_estocastico", dataset, estrategia,
                            metodo, comite, sep = "|")))
}

# ------------------------------------------------------------------
# Auxiliares + fontes (cada GA em seu próprio environment, pois definem
# funções com o mesmo nome e implementações diferentes)
# ------------------------------------------------------------------
bm <- function(partition) outer(partition, partition, FUN = "==") * 1
fobj <- function(x, CO_matrix, B) NA_real_

env_brkga <- new.env(); sys.source(file.path(SRC_DIR, "BRKGA-CE.R"), envir = env_brkga)
env_ag8   <- new.env(); sys.source(file.path(SRC_DIR, "GA-CE.R"),    envir = env_ag8)

PARAMS_BRKGA <- list(pop_size = 50, pe_perc = 0.25, pm_perc = 0.30,
                     rho = 0.6, GEN_MAX = 200, nn = 10)
PARAMS_AG8   <- list(pop_size = 100, pe_perc = 0.25, pm_perc = 0.10,
                     rho = 0.8, GEN_MAX = 200, threshold = 0.7, nn = 20)

METODOS_EST <- list(
  BRKGA_CE = function(PI, k) env_brkga$BRKGA_CE(
    pop_size = PARAMS_BRKGA$pop_size, pe_perc = PARAMS_BRKGA$pe_perc,
    pm_perc = PARAMS_BRKGA$pm_perc, rho = PARAMS_BRKGA$rho,
    PI = PI, k_fixed = k, GEN_MAX = PARAMS_BRKGA$GEN_MAX, nn = PARAMS_BRKGA$nn),
  AG8 = function(PI, k) env_ag8$AG8(
    pop_size = PARAMS_AG8$pop_size, pe_perc = PARAMS_AG8$pe_perc,
    pm_perc = PARAMS_AG8$pm_perc, rho = PARAMS_AG8$rho,
    PI = PI, k_fixed = k, GEN_MAX = PARAMS_AG8$GEN_MAX,
    threshold = PARAMS_AG8$threshold, nn = PARAMS_AG8$nn)
)



meta_part   <- fromJSON(file.path(PART_DIR, "_metadata.json"))
resumo      <- read.csv(file.path(PART_DIR, "_resumo_datasets.csv"), stringsAsFactors = FALSE)
N_COMITES   <- meta_part$n_comites

# Ordem fixa de processamento (igual ao notebook do RKO)
ORDEM_ESTRATEGIAS <- c("hierarquica", "nao_hierarquica", "mistura")
stopifnot(setequal(ORDEM_ESTRATEGIAS, meta_part$estrategias))
ESTRATEGIAS <- ORDEM_ESTRATEGIAS

# ------------------------------------------------------------------
# Configuração (inclui a seed master e os parâmetros)
# ------------------------------------------------------------------
config <- list(r_master_seed = R_MASTER_SEED,
               comites_master_seed = meta_part$comites_master_seed,
               n_comites = N_COMITES, estrategias = ESTRATEGIAS,
               metodos = names(METODOS_EST),
               params_brkga = PARAMS_BRKGA, params_ag8 = PARAMS_AG8,
               nmi_variant = "max",
               esquema_seed = "set.seed(hash_str('MASTER|lit_estocastico|base|estrategia|metodo|comite'))",
               r_version = R.version.string)
cfg_cmp <- function(x) toJSON(x[setdiff(names(x), "r_version")], auto_unbox = TRUE)
if (file.exists(CONFIG_JSON)) {
  antiga <- fromJSON(CONFIG_JSON)
  if (!identical(cfg_cmp(antiga), cfg_cmp(config)))
    stop("Configuração incompatível com a execução anterior (seed, parâmetros ou comitês). ",
         "Use outros arquivos de saída.")
} else {
  write(toJSON(config, auto_unbox = TRUE, pretty = TRUE), CONFIG_JSON)
}
cat(sprintf("R_MASTER_SEED: %d\n", R_MASTER_SEED))

# ------------------------------------------------------------------
# Checkpoint / resume
# ------------------------------------------------------------------
COLS  <- c("dataset", "estrategia", "metodo", "comite", "k", "ARI", "NMI", "tempo_s", "seed")
CHAVE <- c("dataset", "estrategia", "metodo", "comite")
chave <- function(d, e, m, c) paste(d, e, m, c, sep = "|")

ja_feito <- character(0)
if (file.exists(OUT_CSV)) {
  bruto <- read.csv(OUT_CSV, stringsAsFactors = FALSE)
  limpo <- bruto[!is.na(bruto$ARI), ]
  limpo <- limpo[!duplicated(limpo[, CHAVE], fromLast = TRUE), ]
  if (nrow(limpo) < nrow(bruto)) {
    cat(sprintf("Removendo %d linha(s) inválidas/duplicadas do checkpoint\n", nrow(bruto) - nrow(limpo)))
    write.csv(limpo, OUT_CSV, row.names = FALSE)
  }
  ja_feito <- chave(limpo$dataset, limpo$estrategia, limpo$metodo, limpo$comite)
}
cat(sprintf("Execuções já concluídas: %d\n", length(ja_feito)))

anexar <- function(df, path) {
  existe <- file.exists(path)
  write.table(df, path, sep = ",", row.names = FALSE,
              col.names = !existe, append = existe)
}

avaliar <- function(y_true, pi_hat) {
  if (is.null(pi_hat) || any(is.na(pi_hat)) || length(pi_hat) != length(y_true))
    return(list(ari = NA_real_, nmi = NA_real_))
  list(ari = mclust::adjustedRandIndex(y_true, pi_hat),
       nmi = aricode::NMI(y_true, pi_hat, variant = "max"))
}

# ------------------------------------------------------------------
# Loop: base -> estratégia -> comitê -> método
# ------------------------------------------------------------------
total <- nrow(resumo) * length(ESTRATEGIAS) * N_COMITES * length(METODOS_EST)
cat(sprintf("Total de execuções: %d\n", total))
contador <- length(ja_feito)

for (i in seq_len(nrow(resumo))) {
  dataset <- resumo$dataset[i]
  k       <- resumo$k[i]
  y_true  <- read.csv(file.path(PART_DIR, paste0(dataset, "_ground_truth.csv")))$y_true
  
  for (estrategia in ESTRATEGIAS) {
    for (comite in seq_len(N_COMITES)) {
      
      pendentes <- names(METODOS_EST)[!chave(dataset, estrategia, names(METODOS_EST), comite) %in% ja_feito]
      if (length(pendentes) == 0) next
      
      pi_path <- file.path(PART_DIR, dataset, estrategia, sprintf("c%02d_PI.csv", comite))
      PI <- as.matrix(read.csv(pi_path))
      storage.mode(PI) <- "integer"
      
      for (metodo in pendentes) {
        seed <- seed_lit(dataset, estrategia, metodo, comite)
        set.seed(seed)
        
        t0 <- Sys.time()
        pi_hat <- tryCatch(
          METODOS_EST[[metodo]](PI, k),
          error = function(e) {
            anexar(data.frame(dataset = dataset, estrategia = estrategia, metodo = metodo,
                              comite = comite, seed = seed,
                              erro = gsub("\n", " | ", e$message)), ERROS_CSV)
            message(sprintf("[ERRO] %s|%s|c%02d|%s: %s", dataset, estrategia, comite, metodo, e$message))
            NULL
          })
        tempo <- as.numeric(Sys.time() - t0, units = "secs")
        m <- avaliar(y_true, pi_hat)
        if (is.na(m$ari)) next
        
        contador <- contador + 1
        cat(sprintf("[%d/%d] %s | %s | c%02d | %-8s ARI=%.4f NMI=%.4f (%.1fs)\n",
                    contador, total, dataset, estrategia, comite, metodo, m$ari, m$nmi, tempo))
        
        anexar(data.frame(dataset = dataset, estrategia = estrategia, metodo = metodo,
                          comite = comite, labels = paste(pi_hat, collapse = " ")), LABELS_CSV)
        anexar(data.frame(dataset = dataset, estrategia = estrategia, metodo = metodo,
                          comite = comite, k = k, ARI = m$ari, NMI = m$nmi,
                          tempo_s = round(tempo, 4), seed = seed)[, COLS], OUT_CSV)
      }
    }
  }
}
cat(sprintf("\nFinalizado. Resultados em %s\n", OUT_CSV))