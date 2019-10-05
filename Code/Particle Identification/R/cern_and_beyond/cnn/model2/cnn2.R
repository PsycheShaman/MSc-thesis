args = commandArgs(TRUE)
dirX = args[1]

require(keras,lib.loc="/scratch/vljchr004/")

print("wrangling data")

wrangle <- function(dirX){
  
  require(jsonlite,lib.loc="/scratch/vljchr004/")
  
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

####################################################

print("processing layers")

layer_cake <- function(layer){

print(layer)  

  cake_slice <- sapply(dat,`[[`,layer)

n <- which(as.vector(sapply(cake_slice, is.null)))

d <- which(as.vector(sapply(cake_slice, typeof))!="integer")

for(i in n){
  
  cake_slice[[i]] <- matrix(data=0,nrow=17,ncol=24)
  
}

for(i in d){
  
  cake_slice[[i]] <- matrix(data=0,nrow=17,ncol=24)
  
}

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

layer0 <- layer_cake("layer 0")
layer1 <- layer_cake("layer 1")
layer2 <- layer_cake("layer 2")
layer3 <- layer_cake("layer 3")
layer4 <- layer_cake("layer 4")
layer5 <- layer_cake("layer 5")

####################################

x <- array(dim=c(6,24))

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

print(dim(x))

x <- array_reshape(x,dim=c(length(dat),6,24))

print(dim(x))

y <- sapply(dat,`[[`,"pdgCode")

y <- ifelse(abs(y)==11,1,0)

y <- to_categorical(as.vector(as.numeric(y)))

set.seed(1234321)
print("seed set, sampling train indices")
train_ind <- sample(1:nrow(x),size=round(.75*nrow(x)))

train_x <- x[train_ind,,]
train_y <- y[train_ind,]

test_x <- x[-train_ind,,]
test_y <- y[-train_ind,]

model <- keras_model_sequential()

model %>%
  

  layer_conv_1d(
    filter = 24, kernel_size = 3, padding = "causal", 
    input_shape = c(6,24)
  ) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_1d(filter = 24, kernel_size = 3) %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_1d(pool_size = c(2)) %>%
  layer_dropout(0.25) %>%
  
  
  layer_conv_1d(filter = 24, kernel_size = 3, padding = "causal") %>%
  layer_activation("relu") %>%
  layer_conv_1d(filter = 24, kernel_size = 3) %>%
  layer_activation("relu") %>%
  
  # Use max pooling once more
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
 
  layer_dense(2) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

summary(model)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

history <- model %>% fit(
  train_x, train_y,
  batch_size = 128,
  epochs = 100,
  validation_data = list(test_x, test_y),
  shuffle = TRUE
)

png("/scratch/vljchr004/cnn2.png")
plot(history)
dev.off()

p <- predict_proba(model,test_x)

#save_model_hdf5(model,filepath="/scratch/vljchr004/cnn1.h5")

write.csv(p,file="/scratch/vljchr004/cnn1_preds.csv")
write.csv(test_y,file="scratch/vljchr004/test_y.csv")

print("done")
