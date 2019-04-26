#clear R session
rm(list=ls())

#function to read in json files, extract layers, keep only particles in the [2,4)
#GeV range and remove tracks that have any layers with no signal

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

#function to remove tracks that have any pads with zero-sum signal
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

#an anomaly detection and removal algorithm
dodge_rm <- function(dat){
  
  layer0 <- lapply(dat, `[[`,"layer 0")
  
  rs0 <- matrix(nrow=length(dat),ncol=nrow(layer0[[1]]))
  
  for(i in 1:length(layer0)){
    rs0i <- rowSums(layer0[[i]])
    
    rs0[i,] <- rs0i
    
  }
  
  rs0 <- data.frame(rs0)
  
  rs0$nonzero_elements <- 0
  
  for(i in 1:nrow(rs0)){
    
    rs0$nonzero_elements[i] <- length(which(rs0[i,1:17]!=0))
    rs0$sum[i] <- sum(rs0[i,1:17])
    
  }
  
  
  
  layer1 <- lapply(dat, `[[`,"layer 1")
  
  rs1 <- matrix(nrow=length(dat),ncol=nrow(layer1[[1]]))
  
  for(i in 1:length(layer1)){
    rs1i <- rowSums(layer1[[i]])
    
    rs1[i,] <- rs1i
    
  }
  
  rs1 <- data.frame(rs1)
  
  rs1$nonzero_elements <- 0
  
  for(i in 1:nrow(rs1)){
    
    rs1$nonzero_elements[i] <- length(which(rs1[i,1:17]!=0))
    rs1$sum[i] <- sum(rs1[i,1:17])
    
  }
  
  
  
  
  layer2 <- lapply(dat, `[[`,"layer 2")
  
  rs2 <- matrix(nrow=length(dat),ncol=nrow(layer2[[1]]))
  
  for(i in 1:length(layer2)){
    rs2i <- rowSums(layer2[[i]])
    
    rs2[i,] <- rs2i
    
  }
  
  rs2 <- data.frame(rs2)
  
  rs2$nonzero_elements <- 0
  
  for(i in 1:nrow(rs2)){
    
    rs2$nonzero_elements[i] <- length(which(rs2[i,1:17]!=0))
    rs2$sum[i] <- sum(rs2[i,1:17])
  }
  
  
  
  
  layer3 <- lapply(dat, `[[`,"layer 3")
  
  rs3 <- matrix(nrow=length(dat),ncol=nrow(layer3[[1]]))
  
  for(i in 1:length(layer3)){
    rs3i <- rowSums(layer3[[i]])
    
    rs3[i,] <- rs3i
    
  }
  
  rs3 <- data.frame(rs3)
  
  rs3$nonzero_elements <- 0
  
  for(i in 1:nrow(rs3)){
    
    rs3$nonzero_elements[i] <- length(which(rs3[i,1:17]!=0))
    rs3$sum[i] <- sum(rs3[i,1:17])
    
  }
  
  
  
  layer4 <- lapply(dat, `[[`,"layer 4")
  
  rs4 <- matrix(nrow=length(dat),ncol=nrow(layer4[[1]]))
  
  for(i in 1:length(layer4)){
    rs4i <- rowSums(layer4[[i]])
    
    rs4[i,] <- rs4i
    
  }
  
  rs4 <- data.frame(rs4)
  
  rs4$nonzero_elements <- 0
  
  for(i in 1:nrow(rs4)){
    
    rs4$nonzero_elements[i] <- length(which(rs4[i,1:17]!=0))
    rs4$sum[i] <- sum(rs4[i,1:17])
    
  }
  
  
  
  
  layer5 <- lapply(dat, `[[`,"layer 5")
  
  rs5 <- matrix(nrow=length(dat),ncol=nrow(layer5[[1]]))
  
  for(i in 1:length(layer5)){
    rs5i <- rowSums(layer5[[i]])
    
    rs5[i,] <- rs5i
    
  }
  
  rs5 <- data.frame(rs5)
  
  rs5$nonzero_elements <- 0
  
  for(i in 1:nrow(rs5)){
    
    rs5$nonzero_elements[i] <- length(which(rs5[i,1:17]!=0))
    rs5$sum[i] <- sum(rs5[i,1:17])
    
  }
  
  
  rs <- data.frame(cbind(rs0[,18],rs1[,18],rs2[,18],rs3[,18],rs4[,18],rs5[,18]))
  
  rs.v <- rs
  
  for(i in 1:nrow(rs)){
    rs.v[i,] <- as.numeric(scale(as.numeric(rs[i,])))
  }
  
  rs.v <- as.matrix(rs.v)
  
  nans <- apply(rs.v,1,is.nan)
  
  nans <- t(nans)
  
  nans <- apply(nans, 1, all)
  
  nans <- which(nans)
  
  rs.v[nans,] <- 0
  
  # quantile(rs.v,probs=seq(0,1,0.1))
  # 
  # hist(rs.v)
  
  dodge <- apply(rs.v,1,abs)
  
  dodge <- t(dodge)
  
  dodge <- apply(dodge, 1, `>=`,"1.75")
  
  dodge <- t(dodge)
  
  dodge <- apply(dodge,1,any)
  # 
  # length(which(dodge))
  # 
  # length(dodge)
  
  rs.v <- data.frame(cbind(rs.v,dodge))
  
  ##############
  
  rs <- data.frame(cbind(rs0[,19],rs1[,19],rs2[,19],rs3[,19],rs4[,19],rs5[,19]))
  
  for(i in 1:nrow(rs)){
    rs[i,] <- as.numeric(scale(as.numeric(rs[i,])))
  }
  
  rs <- as.matrix(rs)
  
  nans <- apply(rs,1,is.nan)
  
  nans <- t(nans)
  
  nans <- apply(nans, 1, all)
  
  nans <- which(nans)
  
  rs[nans,] <- 0
  
  quantile(rs,probs=seq(0,1,0.1))
  
  hist(rs)
  
  dodge <- apply(rs,1,abs)
  
  dodge <- t(dodge)
  
  dodge <- apply(dodge, 1, `>=`,"1.75")
  
  dodge <- t(dodge)
  
  dodge <- apply(dodge,1,any)
  
  length(which(dodge))
  
  length(dodge)
  
  dodge <- ifelse(dodge,1,0)
  
  rs.v$dodge2 <- dodge
  
  rs.v <- rs.v[,-c(1:6)]
  rs.v$dodgy <- rs.v$dodge+rs.v$dodge2
  
  rs <- cbind(rs0[,19],rs1[,19],rs2[,19],rs3[,19],rs4[,19],rs5[,19])
  
  m <- mean(rs)
  s <- sd(rs)
  
  rs <- (rs-m)/s
  
  # hist(rs,breaks = 1000)
  # abline(v=c(1.75,3,5),col="red")
  
  rs <- apply(rs,1,abs)
  rs <- apply(rs,1,`>=`,"3")
  rs <- apply(rs, 1, any)
  rs <- ifelse(rs,1,0)
  
  rs.v$dodge3 <- rs
  
  rs.v$dodgy <- rs.v$dodgy+rs.v$dodge3
  
  hist(rs.v$dodgy,breaks=c(0:3))
  # length(which(rs.v$dodgy==0))
  # length(which(rs.v$dodgy==1))
  # length(which(rs.v$dodgy==2))
  # length(which(rs.v$dodgy==3))
  
  rs.v <- rs.v[,c(1,2,4,3)]
  
  d <- which(rs.v$dodgy>=2)
  
  dat <- dat[-d]
  
  return(dat)
  
  
}

#function that calls the above three functions recursively by looping through all files in specified directory
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



















