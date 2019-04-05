rm(list=ls())

process <- function(f){
  
  require(jsonlite)
  
  j <- fromJSON(f)
  
  Pt <- sapply(j, `[[`,"PT")
  
  Pt <- as.numeric(Pt)
  
  keep <- which(Pt>=2&Pt<=3)
  
  dont_keep <- 1:length(Pt)
  
  dont_keep <- dont_keep[-keep]
  
  for(i in dont_keep){
    j[[i]] <- NULL
  }
  
  return(j)
  
}

wrangle <- function(run){
  
  require(jsonlite)
  
  files <- list.files(path=paste0("/Users/gerhard/msc-thesis-data/processed/000",run),
                      pattern="*json",
                      full.names=T,
                      recursive=TRUE)
  
  j <- process(files[1])
  
  for(i in 2:length(files)){
    print(paste0("FILE: ",i))
    
    f <- process(files[i])
    j <- c(j,f)
  }
  
  j
  
}

dat265377 <- wrangle("265377")

  
a <- as.integer(rep(0,17*24))
dim(a) <- c(17,24)
  
l <- sapply(dat265377, `[[`,"layer 0")
n <- as.numeric(which(sapply(l,is.null)))

for(i in n){
  dat265377[[i]]$`layer 0` <- a
  
}


l <- sapply(dat265377, `[[`,"layer 1")
n <- as.numeric(which(sapply(l,is.null)))

for(i in n){
  dat265377[[i]]$`layer 1` <- a
}


l <- sapply(dat265377, `[[`,"layer 2")
n <- as.numeric(which(sapply(l,is.null)))

for(i in n){
  dat265377[[i]]$`layer 2` <- a
}


l <- sapply(dat265377, `[[`,"layer 3")
n <- as.numeric(which(sapply(l,is.null)))

for(i in n){
  dat265377[[i]]$`layer 3` <- a
}

l <- sapply(dat265377, `[[`,"layer 4")
n <- as.numeric(which(sapply(l,is.null)))

for(i in n){
  dat265377[[i]]$`layer 4` <- a
}


l <- sapply(dat265377, `[[`,"layer 5")
n <- as.numeric(which(sapply(l,is.null)))

for(i in n){
  dat265377[[i]]$`layer 5` <- a
}


l <- lapply(dat265377, `[[`,"layer 0")
d <- as.numeric(which(sapply(sapply(l,dim),is.null)))

for(i in d){
  dat265377[[i]]$`layer 0` <- a
}


l <- lapply(dat265377, `[[`,"layer 1")
d <- as.numeric(which(sapply(sapply(l,dim),is.null)))

for(i in d){
  dat265377[[i]]$`layer 2` <- a
}

l <- lapply(dat265377, `[[`,"layer 2")
d <- as.numeric(which(sapply(sapply(l,dim),is.null)))

for(i in d){
  dat265377[[i]]$`layer 2` <- a
}

l <- lapply(dat265377, `[[`,"layer 3")
d <- as.numeric(which(sapply(sapply(l,dim),is.null)))

for(i in d){
  dat265377[[i]]$`layer 3` <- a
}

l <- lapply(dat265377, `[[`,"layer 4")
d <- as.numeric(which(sapply(sapply(l,dim),is.null)))

for(i in d){
  dat265377[[i]]$`layer 4` <- a
}


l <- lapply(dat265377, `[[`,"layer 5")
d <- as.numeric(which(sapply(sapply(l,dim),is.null)))

for(i in d){
  dat265377[[i]]$`layer 5` <- a
}

####################################
rm(a,l,d,i,n,z)
####################################

x <- list()

for(i in 1:length(dat265377)){
  
  print(100*(i/length(dat265377)))
  
  l0 <- matrix(dat265377[[i]]$`layer 0`,17,24)
  l1 <- matrix(dat265377[[i]]$`layer 1`,17,24)
  l2 <- matrix(dat265377[[i]]$`layer 2`,17,24)
  l3 <- matrix(dat265377[[i]]$`layer 3`,17,24)
  l4 <- matrix(dat265377[[i]]$`layer 4`,17,24)
  l5 <- matrix(dat265377[[i]]$`layer 5`,17,24)
  
  x. <- array(c(l0,l1,l2,l3,l4,l5),dim=c(17,24,6))
  
  x <- c(x,x.)
  
}

save(x,file="265377.rdata")


