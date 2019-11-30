dirX="C:/Users/gerhard/Documents/msc-thesis-data/processed/000265309"

layer_cake <- function(layer,dat){
  
  print(layer)  
  
  cake_slice <- sapply(dat,`[[`,layer)
  
  n <- which(as.vector(sapply(cake_slice, is.null)))
  
  d <- which(as.vector(sapply(cake_slice, typeof))!="integer")
  
  # for(i in n){
  #   
  #   cake_slice[[i]] <- matrix(data=0,nrow=17,ncol=24)
  #   
  # }
  # 
  # for(i in d){
  #   
  #   cake_slice[[i]] <- matrix(data=0,nrow=17,ncol=24)
  #   
  # }
  
  miss <- unique(c(n,d))
  
  for(i in 1:length(cake_slice)){
    
    a <- as.matrix(cake_slice[[i]])
    
    a <- colSums(a)
    
    if(all(a==0)){
      
      a <- numeric(24)
      
      cake_slice[[i]] <- a
      next()
    }
    
    names(a) <- 1:24
    
    beg <- min(which(a!=0))
    
    end <- min(24,min(which(a[beg:24]==0))+beg-2)
    
    sig1 <- a[beg:end]
    
    sig2 <- a[-c(1:end)]
    
    sig2 <- sig2[!sig2==0]
    
    if(length(sig1)>length(sig2)){
      
      signal <- sig1
      
    }else if(length(sig1==sig2)){
      if(sum(sig1)>=sum(sig2)){
        signal <- sig1
      }else{
        signal <- sig2
      }
    }else{
      signal <- sig2
    }
    
    nm <- names(signal)
    
    nm <- as.numeric(nm)
    
    sig <- numeric(length(a))
    
    sig[nm] <- signal
    
    cake_slice[[i]] <- sig
    
    
  }
  
  cake_slice <- matrix(unlist(cake_slice),ncol=24,byrow=T)
  
  return(cake_slice)
}

wrangle <- function(dirX){
  
  require(jsonlite)
  
  files <- list.files(path=paste0(dirX),
                      pattern="*json",
                      full.names=T,
                      recursive=TRUE)
  
  j <- fromJSON(files[1])
  
  print("---------------------------------------------------------------------------------")
  print(paste0("Processing jsons to list:"))
  
  L <- length(files)
  
  for(i in 2:L){
    print(100*(i/L))
    
    f <- fromJSON(files[i])
    j <- c(j,f)
  }
  
  j
  
}

dat <- wrangle(dirX)
test_dat <- wrangle(dirTest)




layer0 <- layer_cake("layer 0",dat)
layer1 <- layer_cake("layer 1",dat)
layer2 <- layer_cake("layer 2",dat)
layer3 <- layer_cake("layer 3",dat)
layer4 <- layer_cake("layer 4",dat)
layer5 <- layer_cake("layer 5",dat)

t.layer0 <- layer_cake("layer 0",test_dat)
t.layer1 <- layer_cake("layer 1",test_dat)
t.layer2 <- layer_cake("layer 2",test_dat)
t.layer3 <- layer_cake("layer 3",test_dat)
t.layer4 <- layer_cake("layer 4",test_dat)
t.layer5 <- layer_cake("layer 5",test_dat)

####################################

x <- array(dim=c(6,24))
x_t <- array(dim=c(6,24))

print("creating X....")

require(abind,lib.loc="/scratch/vljchr004/")

for(i in 1:length(dat)){

r1 <- as.numeric(unlist(layer0[i,]))
r2 <- as.numeric(unlist(layer1[i,]))
r3 <- as.numeric(unlist(layer2[i,]))
r4 <- as.numeric(unlist(layer3[i,]))
r5 <- as.numeric(unlist(layer4[i,]))
r6 <- as.numeric(unlist(layer5[i,]))

m <- rbind(r1,r2,r3,r4,r5,r6)

m <- array(data=m,dim=c(6,24))

x <- abind(x,m,along=3)
print(i/nrow(layer0))
}

x <- x[,,-1]

for(i in 1:length(test_dat)){

r1 <- as.numeric(unlist(t.layer0[i,]))
r2 <- as.numeric(unlist(t.layer1[i,]))
r3 <- as.numeric(unlist(t.layer2[i,]))
r4 <- as.numeric(unlist(t.layer3[i,]))
r5 <- as.numeric(unlist(t.layer4[i,]))
r6 <- as.numeric(unlist(t.layer5[i,]))

m <- rbind(r1,r2,r3,r4,r5,r6)

m <- array(data=m,dim=c(6,24))

x_t <- abind(x_t,m,along=3)
print(i/nrow(t.layer0))
}

x_t <- x_t[,,-1]

print(dim(x))

zero_check <- function(x){
  
  zeros <- c()
  
  for(i in 1:dim(x)[3]){
    print(sum(x[,,i]))
    if(sum(x[,,i])==0){
      print(sum(x[,,i])==0)
      zeros <- c(zeros,i)
    }
    
  }
  
  x[,,-zeros]
  
  
}

x <- zero_check(x)

x <- array_reshape(x,dim=c(length(dat),6,24))

print(dim(x))

y <- sapply(dat,`[[`,"pdgCode")


y <- ifelse(abs(y)==11,1,0)

y <- y[-zeros]

y <- to_categorical(as.vector(as.numeric(y)))


x_t <- zero_check(x_t)

x_t <- array_reshape(x_t,dim=c(length(test_dat),6,24))

print(dim(x_t))

y_t <- sapply(test_dat,`[[`,"pdgCode")


y_t <- ifelse(abs(y_t)==11,1,0)

y_t <- y_t[-zeros]

y_t <- to_categorical(as.vector(as.numeric(y_t)))


set.seed(1234321)
print("seed set, sampling validation indices")
valid_ind <- sample(1:nrow(x),size=round(.75*nrow(x)))

train_x <- x[train_ind,,]
train_y <- y[train_ind,]

valid_x <- x[-train_ind,,]
valid_y <- y[-train_ind,,]

save(train_x,file="/scratch/vljchr004/data/rdata/six_by_24_by_1/train_x")
save(train_y,file="/scratch/vljchr004/data/rdata/six_by_24_by_1/train_y") 
save(valid_x,file="/scratch/vljchr004/data/rdata/six_by_24_by_1/valid_x") 
save(valid_y,file="/scratch/vljchr004/data/rdata/six_by_24_by_1/valid_y") 
save(x_t,file="/scratch/vljchr004/data/rdata/six_by_24_by_1/test_x") 
save(y_t,file="/scratch/vljchr004/data/rdata/six_by_24_by_1/test_y") 
