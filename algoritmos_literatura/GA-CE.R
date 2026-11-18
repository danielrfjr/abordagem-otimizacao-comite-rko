################################################################################
################################################################################
###
### AG8
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


### SOLUÇÃO
solutionm <- function(s, n, k)
{
  
  x = ceiling(s[1:k]*n)
  
  x = as.integer(x)
  
  if(length(unique(x)) < k)
  {
    x = 1:k
    x = as.integer(x)
  }
  
  return(x)
}




### FUNÇÃO OBJETIVO: K-MEDOIDS + Penalty
fobjm <- function(x, PI_matrix, B, threshold)
{
  n = length(PI_matrix[,1])
  kmax = length(x)
  pi = rep(0, n)
  pi_ok = rep(0, n)
  
  
  ### Cluster: objects to Medoids (threshold)
  for(i in 1:n)
  {
    fit_best = -1
    
    for(k in 1:kmax)
    {
      fit_try = PI_matrix[i, x[k]]
      if(fit_best < fit_try)
      {
        k_best = k
        fit_best = fit_try
      }
    }
    
    
    pi[i] = k_best
  }
  
  ### Cluster: Medoids
  for(k in 1:kmax)
  {
    pi[x[k]] = k
  }
  
  ### Original Silhouette
  d2 = 1 - PI_matrix/B
  sil = mean(silhouette(pi, dmatrix = d2)[(2*n+1):(3*n)])
  
  fitness = sil
  
  # problema de maximização (BRKGA minimiza fitness)
  fitness = (-1) * fitness
  return(fitness)
}




### DECODE
decodem <- function(s, PI_matrix, B, k)
{
  ### CHROMOSOME 's' -> SOLUTION 'x'
  n = length(PI_matrix[,1])
  x = solutionm(s, n, k)
  threshold = s[k+1]
  
  ### FITNESS
  fitness = fobjm(x, PI_matrix, B, threshold)
  
  ### RETURN
  return(fitness)
}




### Genetic Operator - Mutant
mutant <- function(N)
{
  s = runif(N)
  
  return(s)
}




### Genetic Operator - Crossover
crossoverm <- function(s1, s2, rho)
{
  N = length(s1)
  s_off = s1
  
  coin = runif(N)
  s2_win = which(rho < coin)
  s_off[s2_win] = s2[s2_win]
  
  if(identical(s_off,s1) == TRUE)
  {
    s2_win = sample(N, 1)
    s_off[s2_win] = s2[s2_win]
  }
  
  return(s_off)
}




### BRKGA Cluster Ensemble k-medoids
AG8 <- function(pop_size = 20, pe_perc = 0.2, pm_perc = 0.2, rho = 0.8, 
                     PI, k_fixed = 2, GEN_MAX = 100, threshold = 0.1, par=FALSE, nn)
{
  
  ### Objetos e Tamanho do Comitê
  n = dim(PI)[1]
  B = dim(PI)[2]
  
  # Tamanho do Cromossomo
  N = k_fixed+1
  
  # População
  p = pop_size
  pe = ceiling(p*pe_perc)
  pm = ceiling(p*pm_perc)
  pc = p - pe - pm
  
  # Fitness
  fitness = rep(0, p)
  fitness_new = fitness
  
  ### Estrutura de Dados (Matriz de Consenso): PI -> PI_max
  PI_matrix = data_structure_PI(PI, B, n)
  d2 = 1 - PI_matrix/B
  
  
  
  
  ###
  ### Population 1
  ###
  
  P = matrix(runif(p*N), nrow=p, ncol=N)
  Pnew = P
  
  
  # Fitness
  fitness<-apply(P,1,function(Px)decodem(Px, PI_matrix, B, k_fixed))
  
  fitness_best = min(fitness)
  cat("\nGEN 0: ", fitness_best)
  
  
  
  
  ###
  ### MAIN LOOP
  ###
  for(GEN in 1:GEN_MAX)
  {
    
    ### GO1 - ELITE
    # Minimization Problem
    order_elite = order(fitness)[1:pe]
    Pnew[1:pe,] = P[order_elite,]
    
    
    ### GO2 - MUTANT
    Px<-t(apply(as.matrix(rep(N,pm)),1,function(Nx) mutant(Nx)))
    Pnew[(pe+1):(pe+pm),]<-Px
    
    
    ### GO3 - CROSSOVER
    for(i in (pe+pm+1):p)
    {
      # Select Chromosomes
      s1 = P[sample(1:pe, 1),]
      s2 = P[sample((pe+1):p, 1),]
      
      # Crossover
      s_off = crossoverm(s1, s2, rho)
      
      # Offspring
      Pnew[i,] = s_off
    }
    
    
    
    
    ### NEW POPULATION
    fitness_new[1:pe] = fitness[order_elite]
    fxe<-apply(Pnew[(pe+1):p,],1,function(Px)decodem(Px, PI_matrix, B, k_fixed))
    
    fitness_new[(pe+1):p]<-fxe
    
    
    P = Pnew
    fitness = fitness_new
    
    i_best = order(fitness)[1]
    s_best = P[i_best,]
    
    if(fitness_best != fitness[i_best])
    {
      fitness_best = fitness[i_best]
      cat("\nGEN ", GEN, ": ", fitness_best)
    }
    
  }
  
  
  
  
  ### RETURN
  cat("\nFINAL: ", fitness_best)
  x = solutionm(s_best, n, k_fixed)
  #pi_best = partition(x, PI_matrix, B, threshold = threshold, nn)
  #fit_best = mean(silhouette(pi_best, dmatrix = d2)[(2*n+1):(3*n)])
  pi_best = NULL
  fit_best = -2
  cat("\nmedoids: ", x, "\n\n")
  
  order_elite = order(fitness)[1:pe]
  Pnew[1:pe,] = P[order_elite,]
  
  for(i in order_elite)
  {
    x = solutionm(P[i,], n, k_fixed)
    #cat("\nmedoids: ", x)
    pi_1 = partition(x, PI_matrix, B, threshold = threshold, nn)
    sil = mean(silhouette(pi_1, dmatrix = d2)[(2*n+1):(3*n)])
    #cat("\nsilhouette: ", sil)
    if(fit_best < sil)
    {
      pi_best = pi_1
      fit_best = sil
      #cat("\nnew best silhouette: ", sil)
    }
  }
  
  cat("\nbest silhouette: ", fit_best)
  
  return(pi_best)
}





#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


partition <- function(x, PI_matrix, B, threshold, nn)
{
  n = length(PI_matrix[,1])
  kmax = length(x)
  pi = rep(0, n)
  pi_0 = rep(0, n)
  d2 = 1 - PI_matrix/B
  
  ### Cluster: objects to Medoids (threshold)
  for(i in 1:n)
  {
    fit_best = -1
    
    for(k in 1:kmax)
    {
      fit_try = PI_matrix[i, x[k]]
      if(fit_best < fit_try)
      {
        k_best = k
        fit_best = fit_try
      }
    }
    
    if(threshold*B <= fit_best) 
    {
      pi[i] = k_best
    }
    pi_0[i] = k_best
  }
  
  ### Cluster: Medoids
  for(k in 1:kmax)
  {
    pi[x[k]] = k
    pi_0[x[k]] = k
  }
  
  ### Passo 2
  # ordenar objetos pela proximidade com medoid
  # realocar eles conforme votação dos 5 vizinhos mais proximos
  d2_m = rep(0, n)
  for(i in 1:n)
  {
    d2_m[i] = d2[i, x[pi_0[i]]]
  }
  ord_i = order(d2_m, decreasing = FALSE)
  #cat("\npi: ", pi )
  #cat("\npi_0: ", pi_0 )
  #cat("\nd2_m: ", d2_m )
  #cat("\nord_i: ", ord_i)
  
  ### Cluster: objects to clusters (no threshold)
  for(i in ord_i)
  {
    if(pi[i] == 0)
    {
      
      #plot(dataset, col = pi)
      #Sys.sleep(0.05)
      #cat("\nfit_best ", fit_best, "; i ", i, "; cluster ", k_best)
      
      fit_best = -1
      
      for(k in 1:kmax)
      {
        objects_k = which(pi == k)
        #fit_try = mean(    PI_matrix[i, objects_k]    )
        # Get the mean of the top 5 max values
        top_values <- tail(sort(PI_matrix[i, objects_k], decreasing = FALSE), nn)
        fit_try <- mean(top_values)
        #fit_try <- median(top_values)
        if(fit_best < fit_try)
        {
          k_best = k
          fit_best = fit_try
        }
      }
      
      #cat("\ni", i)
      #cat("\nk_best", k_best)
      #cat("\nobjects_k:\n", objects_k)
      
      pi[i] = k_best
    }
  }
  
  return(pi)
}









