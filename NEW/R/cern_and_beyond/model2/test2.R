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

#layer.info <- (layer0+layer1+layer2+layer3+layer4+layer5)/6

layer.info <- cbind(layer0,layer1,layer2,layer3,layer4,layer5)

zeros <- which(rowSums(layer.info)==0)

pdg <- sapply(dat,`[[`,"pdgCode")

pdg <- ifelse(abs(pdg)==11,1,0)

layer.info <- layer.info[-zeros,]
pdg <- pdg[-zeros]

######################################################

print("initialize keras")

set.seed(1234321)
print("seed set, sampling train indices")
train_ind <- sample(1:nrow(layer.info),size=round(.75*nrow(layer.info)))

print("y to categorical")

y <- to_categorical(as.vector(as.numeric(pdg)))

print("get train_x")

train_x <- layer.info[train_ind,]

print("get train_y")

train_y <- y[train_ind,]

print("same for test_x")

test_x <- layer.info[-train_ind,]

print("and test_y")

test_y <- y[-train_ind,]

print("build model")

model1 <- keras_model_sequential() %>%

  layer_dense(units = 256,activation = "relu",input_shape = ncol(train_x)) %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=256,activation = "relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=2,activation = "softmax") 

model1 %>% compile(
  optimizer=optimizer_sgd(),
  loss="binary_crossentropy",
  metrics=c("accuracy")
)

history <- model1 %>% fit(train_x,train_y,epochs=20,validation_split=0.2)
model1 %>% save_model_hdf5("/scratch/vljchr004/Rmodel1.h5")

preds <- predict_proba(model1,test_x)

preds <- cbind(preds,test_y)

write.csv(preds,file="/scratch/vljchr004/preds.csv")

png("/scratch/vljchr004/model1_history.png")

plot(history)

dev.off()
