load("x_265377")
load("265377_full")

y <- sapply(dat265377, `[[`,"pdgCode")

y <- ifelse(abs(y)==211,1,0)

x_2 <- matrix(ncol=144,nrow=dim(x)[4])

for(i in 1:dim(x)[4]){
  
  this.i <- x[,,,i]
  
  print(100*(i/dim(x)[4]))
  
  entry <- numeric(24)
  
  for(j in 1:dim(x)[3]){
    
    this.j <- this.i[,,j]
    
    this.entry <- apply(X=this.j,MARGIN = 2,FUN=sum)
    
    entry <- rbind(entry,this.entry)
    
    if(j==dim(x)[3]){
      
      entry <- apply(X=entry, MAR=2, FUN=mean)
      
    }
    
  }
  
  x_2[i,] <- entry 
  
}

require(keras)

y <- to_categorical(y)











