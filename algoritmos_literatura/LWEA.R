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


### ENTROPY H(x)
entropy1 <- function(p)
{
  length(p) = n
  x = length(unique(p))
  H = 0
  
  for(i in 1:x)
  {
    prob = length(which(p==i))/n
    if(prob != 0) {H = H + prob * log2(prob)}
  }
  H = -H
  
  return(H)
}

### JOINT ENTROPY H(x,y)
entropy2 <- function(p1, p2)
{
  length(p1) = n
  
  X = length(unique(p1))
  Y = length(unique(p2))
  
  H = 0
  
  for(x in 1:X)
  {
    for(y in 1:Y)
    {
      prob = length(  intersect(which(p1==x), which(p2==y))  )  /  n
      
      if(prob != 0) {H = H + prob * log2(prob)}
    }
  }
  
  H = -H
  
  return(H)
}


### Entropy H_b(Ci)
Hc <- function(Ci, Cj)
{
  p = length(intersect(Ci, Cj)) / length(Ci)
  
  H = 0
  
  if(p != 0)
  {
    H = p * log2(p)
    
    H = -H
  }
  
  return(H)
}





# Locally Weighted Ensemble Clustering
# Dong Huang
LW_PI <- function(PI, theta)
{
  n = dim(PI)[1]
  B = dim(PI)[2]
  
  ### Construct C list
  # C = {C_1, ..., C_nc}
  # nC = {n1, ..., nB}
  
  nC = rep(0, B)
  for(b in 1:B)
  {
    nC[b] = length(unique(PI[,b]))
  }
  
  nCsum = nC
  for(b in 2:B)
  {
    nCsum[b] = nCsum[b-1] + nCsum[b]
  }
  
  
  C = list()
  C = c(C, 1:nCsum[B])
  
  for(k in (0+1):nCsum[1])
  {
    k_i = k - 0
    C[[k]] = which(PI[,1] == k_i)
  }
  
  for(b in 2:B)
  {
    for(k in (nCsum[b-1]+1):nCsum[b])
    {
      k_i = k - nCsum[b-1]
      C[[k]] = which(PI[,b] == k_i)
    }
  }
  
  
  
  
  ### H_PI(Ci)
  H_PI = rep(0, nCsum[B])
  
  for(i in 1:nCsum[B])
  {
    for(j in 1:nCsum[B])
    {
      H_PI[i] = H_PI[i] + Hc(C[[i]], C[[j]])
    }
  }
  
  
  
  
  ### ECI
  # theta = 0.4 # conforme Dong Huang 2018 - LW EAC
  ECI = rep(0, nCsum[B])
  
  for(i in 1:nCsum[B])
  {
    for(j in 1:nCsum[B])
    {
      ECI[i] = exp(  -H_PI[i] / (theta * B)  )
    }
  }
  
  
  
  
  ### EAC + LW
  PI_matrix = matrix(0, n, n)
  
  
  pi = bm(PI[,1])
  for(k in 1:nCsum[1])
  {
    pi[C[[k]], C[[k]]] = pi[C[[k]], C[[k]]] * ECI[k]
  }
  PI_matrix = PI_matrix + pi
  
  
  for(b in 2:B)
  {
    pi = bm(PI[,b])
    for(k in (nCsum[b-1]+1):nCsum[b])
    {
      pi[C[[k]], C[[k]]] = pi[C[[k]], C[[k]]] * ECI[k]
    }
    PI_matrix = PI_matrix + pi
  }
  
  PI_matrix = PI_matrix/B
  
  
  ### Return
  return(PI_matrix)
}




### Cluster Ensemble EAC
CE_LWEAC <- function(PI, consensus = "average", k_fixed = 2, theta = 0.4)
{
  n = dim(PI)[1]
  B = dim(PI)[2]
  
  # Matriz de Consenso
  #CO_matrix = data_structure_PI(PI, B, n)
  CO_matrix = LW_PI(PI, theta)
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

