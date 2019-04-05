load("265377.rdata")

x. <- x[1]
library(reshape2)
library(rgl)
M=melt(x.)

linMap <- function(x, from, to){
  
  (x - min(x)) / max(x - min(x)) * (to - from) + from
  
}

x.$value <- linMap(x.$value,0,1000)
  

rgl::points3d(M$Var1,M$Var2,M$Var3,size=10,color=heat.colors(1000)[M$value])




