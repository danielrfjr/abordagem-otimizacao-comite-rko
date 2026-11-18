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
CE_EAC <- function(PI, consensus = "average", k_fixed = 2)
{
  n = dim(PI)[1]
  B = dim(PI)[2]
  
  # Matriz de Consenso
  CO_matrix = data_structure_PI(PI, B, n)
  d = as.dist(1 - CO_matrix/B)
  
  
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
  
  return(x_consensus)
}

