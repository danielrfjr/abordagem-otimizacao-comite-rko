################################################################################
################################################################################
###
### Cluster Ensemble - EAC
###
################################################################################
################################################################################


### MATRIZ DE SIMILARIDADES
bmdt <- function(partition, cb)
{
  n = length(partition)
  bm = matrix(rep(0,n*n), n, n)
  
  g = abs(partition[cb[,1]]-partition[cb[,2]])
  
  bm[cb] = 1 - sign(g)
  
  
  return(bm)
}


### DATA STRUCTURE
data_structure_PI <- function(PI, B, n)
{
  cb = t(combn(n,2))
  
  PI_matrix = matrix(0, n, n)
  
  for(b in 1:B)
  {
    pi = bmdt(PI[,b], cb)
    PI_matrix = PI_matrix + pi
  }
  
  PI_matrix = PI_matrix + t(PI_matrix)
  
  PI_matrix[cbind(1:n, 1:n)] = B
  
  return(PI_matrix)
}



### Cluster Ensemble EAC
CE_WEAC <- function(PI, consensus = "average", k_fixed = 2)
{
  n = dim(PI)[1]
  B = dim(PI)[2]
  
  # Matriz de Consenso
  CO_matrix = weight_matrix(PI)
  if( 1 <= sum(is.na(CO_matrix)) ) {stop("\nERRO!!!\nNaN encontrado.")}
  d = as.dist(1 - CO_matrix)
  
  
  # Função Consenso
  # method = "single", "complete", "average", "ward.D2", "median"
  # "pam"
  if(consensus != "pam")
  {
    tx = hclust(d, method = consensus)
    x_consensus = cutree(tx, k = k_fixed)
  } else
  {
    x_consensus = pam(d, k_fixed, metric = "euclidean")[[3]]
  }
  
  fitness = fobj(x_consensus, CO_matrix, B)
  cat("\nMedian Partition(RI): ", 100 * (1 - (fitness/B)/(n*(n-1))), "\n")
  
  return(x_consensus)
}




### Weighted EAC
weight_matrix <- function(PI)
{
  B = dim(PI)[2]
  
  ### CAI(P)
  CAI = rep(0, B)
  
  for(b in 1:B)
  {
    for(b_extra in 1:B)
    {
      if(b != b_extra)
      {
        CAI[b] = CAI[b] + NMI(PI[,b], PI[,b_extra], variant = "sqrt")
      }
    }
  }
  
  CAI = CAI/(B-1)
  if( 1 <= sum(is.na(CAI)) ) {stop("\nERRO!!!\nNaN encontrado.")}
  
  ### NCAI
  NCAI = CAI/max(CAI)
  
  ### I_NCAI
  beta = 2 # conforme Dong Huang 2015 - WEAC
  
  I_NCAI = (NCAI)^beta
  
  ### W
  W = I_NCAI / sum(I_NCAI)
  if( 1 <= sum(is.na(W)) ) {stop("\nERRO!!!\nNaN encontrado.")}
  
  
  
  ###
  ### PI_matrix (corrigido)
  ###
  n = dim(PI)[1]
  cb = t(combn(n,2))
  PI_matrix = matrix(0, n, n)
  
  for(b in 1:B)
  {
    pi = bmdt(PI[,b], cb)
    PI_matrix = PI_matrix + pi * W[b]
  }
  
  PI_matrix = PI_matrix + t(PI_matrix)
  
  PI_matrix[cbind(1:n, 1:n)] = 1
  
  return(PI_matrix)
}