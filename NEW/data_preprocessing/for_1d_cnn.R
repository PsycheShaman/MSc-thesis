rm(list=ls())

process <- function(f){
  
  require(jsonlite)
  
  j <- fromJSON(f)
  
  P <- sapply(j, `[[`,"P")
  
  P <- as.numeric(P)
  
  keep <- which(P>=2&P<4)
  
  dont_keep <- 1:length(P)
  
  dont_keep <- dont_keep[-keep]
  
  l <- sapply(j, `[[`,"layer 0")
  n <- as.numeric(which(sapply(l,is.null)))
  d <- which(as.vector(sapply(l, typeof))!="integer")
  
  l <- sapply(j, `[[`,"layer 1")
  n <- c(n,as.numeric(which(sapply(l,is.null))))
  d <- c(d,which(as.vector(sapply(l, typeof))!="integer"))
  
  l <- sapply(j, `[[`,"layer 2")
  n <- c(n,as.numeric(which(sapply(l,is.null))))
  d <- c(d,which(as.vector(sapply(l, typeof))!="integer"))
  
  l <- sapply(j, `[[`,"layer 3")
  n <- c(n,as.numeric(which(sapply(l,is.null))))
  d <- c(d,which(as.vector(sapply(l, typeof))!="integer"))
  
  l <- sapply(j, `[[`,"layer 4")
  n <- c(n,as.numeric(which(sapply(l,is.null))))
  d <- c(d,which(as.vector(sapply(l, typeof))!="integer"))
  
  l <- sapply(j, `[[`,"layer 5")
  n <- c(n,as.numeric(which(sapply(l,is.null))))
  d <- c(d,which(as.vector(sapply(l, typeof))!="integer"))
  
  dont_keep <- c(dont_keep,n,d)
  
  j <- j[-dont_keep]
  
  return(j)
  
}

remove_zero_pads <- function(dat){
  
  
  layer0 <- lapply(dat, `[[`,"layer 0")
  layer1 <- lapply(dat, `[[`,"layer 1")
  layer2 <- lapply(dat, `[[`,"layer 2")
  layer3 <- lapply(dat, `[[`,"layer 3")
  layer4 <- lapply(dat, `[[`,"layer 4")
  layer5 <- lapply(dat, `[[`,"layer 5")
  
  l0sum <- sapply(layer0, sum)
  l1sum <- sapply(layer1, sum)
  l2sum <- sapply(layer2, sum)
  l3sum <- sapply(layer3, sum)
  l4sum <- sapply(layer4, sum)
  l5sum <- sapply(layer5, sum)
  
  z <- which(l0sum==0)
  z <- c(z,which(l1sum==0))
  z <- c(z,which(l2sum==0))
  z <- c(z,which(l3sum==0))
  z <- c(z,which(l4sum==0))
  z <- c(z,which(l5sum==0))
  
  z <- unique(z)
  
  
  dat <- dat[-z]
  
  return(dat)
  
}

wrangle <- function(run){
  
  require(jsonlite)
  
  files <- list.files(path=paste0("C:/Users/gerhard/Documents/msc-thesis-data/processed/000",run),
                      pattern="*json",
                      full.names=T,
                      recursive=TRUE)
  
  j <- process(files[1])
  
  for(i in 2:length(files)){
    print(paste0("FILE: ",i))
    
    f <- process(files[i])
    j <- c(j,f)
  }
  
  j <- remove_zero_pads(j)
  
  j
  
}

dat265309 <- wrangle("265309")


layer0 <- lapply(dat265309, `[[`,"layer 0")

rs0 <- matrix(nrow=length(dat265309),ncol=nrow(layer0[[1]]))

for(i in 1:length(layer0)){
  rs0i <- rowSums(layer0[[i]])
  
  rs0[i,] <- rs0i
  
}

rs0 <- data.frame(rs0)

rs0$nonzero_elements <- 0

for(i in 1:nrow(rs0)){
  
  rs0$nonzero_elements[i] <- length(which(rs0[i,1:17]!=0))
  
}



layer1 <- lapply(dat265309, `[[`,"layer 1")

rs1 <- matrix(nrow=length(dat265309),ncol=nrow(layer1[[1]]))

for(i in 1:length(layer1)){
  rs1i <- rowSums(layer1[[i]])
  
  rs1[i,] <- rs1i
  
}

rs1 <- data.frame(rs1)

rs1$nonzero_elements <- 0

for(i in 1:nrow(rs1)){
  
  rs1$nonzero_elements[i] <- length(which(rs1[i,1:17]!=0))
  
}




layer2 <- lapply(dat265309, `[[`,"layer 2")

rs2 <- matrix(nrow=length(dat265309),ncol=nrow(layer2[[1]]))

for(i in 1:length(layer2)){
  rs2i <- rowSums(layer2[[i]])
  
  rs2[i,] <- rs2i
  
}

rs2 <- data.frame(rs2)

rs2$nonzero_elements <- 0

for(i in 1:nrow(rs2)){
  
  rs2$nonzero_elements[i] <- length(which(rs2[i,1:17]!=0))
  
}




layer3 <- lapply(dat265309, `[[`,"layer 3")

rs3 <- matrix(nrow=length(dat265309),ncol=nrow(layer3[[1]]))

for(i in 1:length(layer3)){
  rs3i <- rowSums(layer3[[i]])
  
  rs3[i,] <- rs3i
  
}

rs3 <- data.frame(rs3)

rs3$nonzero_elements <- 0

for(i in 1:nrow(rs3)){
  
  rs3$nonzero_elements[i] <- length(which(rs3[i,1:17]!=0))
  
}



layer4 <- lapply(dat265309, `[[`,"layer 4")

rs4 <- matrix(nrow=length(dat265309),ncol=nrow(layer4[[1]]))

for(i in 1:length(layer4)){
  rs4i <- rowSums(layer4[[i]])
  
  rs4[i,] <- rs4i
  
}

rs4 <- data.frame(rs4)

rs4$nonzero_elements <- 0

for(i in 1:nrow(rs4)){
  
  rs4$nonzero_elements[i] <- length(which(rs4[i,1:17]!=0))
  
}




layer5 <- lapply(dat265309, `[[`,"layer 5")

rs5 <- matrix(nrow=length(dat265309),ncol=nrow(layer5[[1]]))

for(i in 1:length(layer5)){
  rs5i <- rowSums(layer5[[i]])
  
  rs5[i,] <- rs5i
  
}

rs5 <- data.frame(rs5)

rs5$nonzero_elements <- 0

for(i in 1:nrow(rs5)){
  
  rs5$nonzero_elements[i] <- length(which(rs5[i,1:17]!=0))
  
}


rs <- cbind(rs0[,18],rs1[,18],rs2[,18],rs3[,18],rs4[,18],rs5[,18])





















































