################################################################################
### Literatura: EAC / WEAC / LWEA  x  SL / AL / CL
### 1 execução por (base, estratégia, comitê, método).
################################################################################

library(cluster)
library(aricode)
library(mclust)
library(jsonlite)

# ------------------------------------------------------------------
# Caminhos
# O script deve ser executado de dentro de algoritmos_literatura/
# (o diretório de trabalho precisa conter EAC.R, WEAC.R e LWEA.R).
# Estrutura do projeto:
#   projeto/
#     algoritmos_literatura/   <- diretório de trabalho (EAC.R, WEAC.R, LWEA.R)
#     results/                 <- partitions_for_R/ (entrada) e CSVs/JSON (saída)
#     datasets/, src/, ...
# ------------------------------------------------------------------
SRC_DIR     <- "."                                     # EAC.R, WEAC.R, LWEA.R estão na pasta atual
RESULTS_DIR <- file.path("..", "results")              # results/ fica um nível acima
PART_DIR    <- file.path(RESULTS_DIR, "partitions_for_R")

OUT_CSV     <- file.path(RESULTS_DIR, "resultados_literatura_eac_weac_lwea.csv")
LABELS_CSV  <- file.path(RESULTS_DIR, "labels_literatura_eac_weac_lwea.csv")
ERROS_CSV   <- file.path(RESULTS_DIR, "erros_literatura_eac_weac_lwea.csv")
CONFIG_JSON <- file.path(RESULTS_DIR, "config_literatura_eac_weac_lwea.json")

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)


# ------------------------------------------------------------------
# Funções auxiliares que faltavam nos arquivos originais
# (definidas ANTES do source, para CE_WEAC/CE_LWEAC as encontrarem)
# ------------------------------------------------------------------
bm <- function(partition) outer(partition, partition, FUN = "==") * 1
fobj <- function(x, CO_matrix, B) NA_real_   # stub: só afeta um print diagnóstico

source(file.path(SRC_DIR, "EAC.R"))    # CE_EAC
source(file.path(SRC_DIR, "WEAC.R"))   # CE_WEAC, weight_matrix
source(file.path(SRC_DIR, "LWEA.R"))   # CE_LWEAC, LW_PI, Hc

METODOS_DET <- list(EAC = CE_EAC, WEAC = CE_WEAC, LWEA = CE_LWEAC)
VARIANTES   <- c(SL = "single", AL = "average", CL = "complete")
NOMES_METODOS <- as.vector(outer(names(METODOS_DET), names(VARIANTES),
                                 function(m, v) paste0(m, "-", v)))   # 9 nomes

meta_part   <- fromJSON(file.path(PART_DIR, "_metadata.json"))
resumo      <- read.csv(file.path(PART_DIR, "_resumo_datasets.csv"), stringsAsFactors = FALSE)

N_COMITES   <- meta_part$n_comites

# Ordem fixa de processamento (igual ao notebook do RKO)
ORDEM_ESTRATEGIAS <- c("hierarquica", "nao_hierarquica", "mistura")
stopifnot(setequal(ORDEM_ESTRATEGIAS, meta_part$estrategias))
ESTRATEGIAS <- ORDEM_ESTRATEGIAS

# ------------------------------------------------------------------
# Configuração (impede retomar com comitês diferentes)
# ------------------------------------------------------------------
config <- list(comites_master_seed = meta_part$comites_master_seed,
               n_comites = N_COMITES, estrategias = ESTRATEGIAS,
               metodos = NOMES_METODOS, nmi_variant = "max")
if (file.exists(CONFIG_JSON)) {
  antiga <- fromJSON(CONFIG_JSON)
  if (!identical(toJSON(antiga, auto_unbox = TRUE), toJSON(config, auto_unbox = TRUE)))
    stop("Configuração incompatível com a execução anterior. Use outros arquivos de saída.")
} else {
  write(toJSON(config, auto_unbox = TRUE, pretty = TRUE), CONFIG_JSON)
}

# ------------------------------------------------------------------
# Checkpoint / resume
# ------------------------------------------------------------------
COLS   <- c("dataset", "estrategia", "metodo", "comite", "k", "ARI", "NMI", "tempo_s")
CHAVE  <- c("dataset", "estrategia", "metodo", "comite")
chave  <- function(d, e, m, c) paste(d, e, m, c, sep = "|")

ja_feito <- character(0)
if (file.exists(OUT_CSV)) {
  bruto  <- read.csv(OUT_CSV, stringsAsFactors = FALSE)
  limpo  <- bruto[!is.na(bruto$ARI), ]                              # falhas serão refeitas
  limpo  <- limpo[!duplicated(limpo[, CHAVE], fromLast = TRUE), ]   # remove duplicatas
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
       nmi = aricode::NMI(y_true, pi_hat, variant = "max"))   # igual ao nmi_max do Python
}

# ------------------------------------------------------------------
# Loop: base -> estratégia -> comitê -> método
# ------------------------------------------------------------------
total <- nrow(resumo) * length(ESTRATEGIAS) * N_COMITES * length(NOMES_METODOS)
cat(sprintf("Total de execuções: %d\n", total))
contador <- length(ja_feito)

for (i in seq_len(nrow(resumo))) {
  dataset <- resumo$dataset[i]
  k       <- resumo$k[i]
  y_true  <- read.csv(file.path(PART_DIR, paste0(dataset, "_ground_truth.csv")))$y_true
  
  for (estrategia in ESTRATEGIAS) {
    for (comite in seq_len(N_COMITES)) {
      
      pendentes <- NOMES_METODOS[!chave(dataset, estrategia, NOMES_METODOS, comite) %in% ja_feito]
      if (length(pendentes) == 0) next
      
      pi_path <- file.path(PART_DIR, dataset, estrategia, sprintf("c%02d_PI.csv", comite))
      PI <- as.matrix(read.csv(pi_path))
      storage.mode(PI) <- "integer"
      
      for (metodo_nome in names(METODOS_DET)) {
        for (variante_nome in names(VARIANTES)) {
          nome_completo <- paste0(metodo_nome, "-", variante_nome)
          if (!nome_completo %in% pendentes) next
          
          t0 <- Sys.time()
          pi_hat <- tryCatch(
            METODOS_DET[[metodo_nome]](PI, consensus = VARIANTES[[variante_nome]], k_fixed = k),
            error = function(e) {
              anexar(data.frame(dataset = dataset, estrategia = estrategia, metodo = nome_completo,
                                comite = comite, erro = gsub("\n", " | ", e$message)), ERROS_CSV)
              message(sprintf("[ERRO] %s|%s|c%02d|%s: %s", dataset, estrategia, comite,
                              nome_completo, e$message))
              NULL
            })
          tempo <- as.numeric(Sys.time() - t0, units = "secs")
          m <- avaliar(y_true, pi_hat)
          if (is.na(m$ari)) next        # não grava: será refeito ao retomar
          
          contador <- contador + 1
          cat(sprintf("[%d/%d] %s | %s | c%02d | %-9s ARI=%.4f NMI=%.4f (%.2fs)\n",
                      contador, total, dataset, estrategia, comite, nome_completo,
                      m$ari, m$nmi, tempo))
          
          # labels primeiro; resultados por último (define "concluído")
          anexar(data.frame(dataset = dataset, estrategia = estrategia, metodo = nome_completo,
                            comite = comite, labels = paste(pi_hat, collapse = " ")), LABELS_CSV)
          anexar(data.frame(dataset = dataset, estrategia = estrategia, metodo = nome_completo,
                            comite = comite, k = k, ARI = m$ari, NMI = m$nmi,
                            tempo_s = round(tempo, 4))[, COLS], OUT_CSV)
        }
      }
    }
  }
}
cat(sprintf("\nFinalizado. Resultados em %s\n", OUT_CSV))