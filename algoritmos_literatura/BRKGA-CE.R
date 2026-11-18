################################################################################
################################################################################
###
### BRKGA - Cluster Ensemble - Medoid
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
  
  x = ceiling(s*n)
  
  x = as.integer(x)
  
  if(length(unique(x)) < k)
  {
    x = 1:k
    x = as.integer(x)
  }
  
  return(x)
}




### FUNÇÃO OBJETIVO: K-MEDOIDS + Penalty
fobjm <- function(x, PI_matrix, B)
{
  n = length(PI_matrix[,1])
  kmax = length(x)
  pi = rep(0, n)
  
  fitness = 0
  
  
  ### Objects to Medoids
  for(i in 1:n)
  {
    empate = 0
    fit_best = -1
    
    for(k in 1:kmax)
    {
      fit_try = PI_matrix[i, x[k]]
      if(fit_best < fit_try)
      {
        k_best = k
        i_best = i
        fit_best = fit_try
      }
    }
    
    pi[i_best] = k_best
    fitness = fitness + fit_best
  }
  #cat("\n",pi,"\n")
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
  
  ### FITNESS
  fitness = fobjm(x, PI_matrix, B)
  
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
BRKGA_CE <- function(pop_size = 20, pe_perc = 0.2, pm_perc = 0.2, rho = 0.8, 
                     PI, k_fixed = 2, GEN_MAX = 100, nn = 10, ls = 1,par=FALSE)
{
  
  ### Objetos e Tamanho do Comitê
  n = dim(PI)[1]
  B = dim(PI)[2]
  
  # Tamanho do Cromossomo
  N = k_fixed
  
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
      
      # Local Search
      #if(i == pe+pm+ls)
      if( (i == pe+pm+ls) && (GEN%%10 == 0) )
      {
        s_off_ls = local_search_medoid(s_off, d2, k_fixed)
        s_off = s_off_ls
      }
      
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
  cat("\nFINAL GEN ", GEN, ": ", fitness_best, "\n")
  
  
  x = solutionm(s_best, n, k_fixed)
  cat("\nmedoids: ", x, "\n")
  
  ### Cluster Ensemble NN
  pi = CE_nn(x, n, PI_matrix, B, nn)
  
  return(pi)
}





#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


sum_med <- function(x, d2)
{
  total = 0
  
  for(i in 1:n)
  {
    total = total + min(d2[i, x])
  }
  
  return(total)
}


local_search_medoid <- function(s, d2, k_max)
{
  n = dim(d2)[1]
  y = solutionm(s, n, k_max)
  
  x_final = cluster::pam(x=d2, diss=TRUE, k=k_max, n=n, medoids=y)$id.med
  
  # Solução x -> Cromossomo s
  s_final = x_final/n
  
  
  # problema de maximização (BRKGA minimiza fitness)
  #fitness = sil
  #fitness = (-1) * fitness
  return(s_final)
  
  
}













#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


CE_nn <- function(x, n, PI_matrix, B, nn)
{
  kmax = length(x)
  PI_matrix = PI_matrix/B
  fitness = 0
  ndsmax = nn
  
  
  pi = rep(0, n)
  pi[x[1:kmax]]<-1:kmax
  
  for(i in 1:(n-kmax))
  {
    fit_best = -1
    
    
    
    for(j in 1:n)
    {
      if(fit_best == 1) {break}
      if(pi[j] == 0)
      {
        
        
        for(k in 1:kmax)
        {
          wch_k = which(pi==k)
          ds = PI_matrix[j, wch_k]
          nds = length(ds)
          nds = min(nds, ndsmax)
          fit_try = mean(  sort(ds, TRUE)[1:nds]  ) # maiores valores [0-1]
          #fit_try = mean(ds)
          
          if(fit_best < fit_try)
          {
            fit_best = fit_try
            k_best = k
            j_best = j
          }
        }
        
        
      }
    }
    
    
    # Cluster k_best absorve Objeto j_best
    pi[j_best] = k_best
  }
  
  
  ### RETURN
  pi = as.integer(pi)
  return(pi)
}




