#required packages
require(jsonlite, lib.loc = "/scratch/vljchr004/")

#command line arguments
args = commandArgs(TRUE)
dirX = args[1]

if(!file.exists(dirX)){
  cat('Cannot find', input_path, 'exiting!\n')
  stop()
}

#json wrangling

wrangle <- function(dirX){
  #get all JSON files in directory specified by command line arguments to RScript
  files <- list.files(path=paste0(dirX),
                      pattern="*json",
                      full.names=T,
                      recursive=TRUE)
  #JSON is read in as an R list object
  j <- fromJSON(files[1])
  
  print("---------------------------------------------------------------------------------")
  print(paste0("Processing jsons to list:"))
  
  #subsequent JSONs are appended to this list
  for(i in 2:length(files)){
    #progress indication to stdout
    print(100*(i/length(files)))
    
    f <- fromJSON(files[i])
    j <- c(j,f)
  }
  
  j
  
}

#use function defined above to wrangle JSONs

dat <- wrangle(dirX)
print("---------------------------------------------------------------------------------")
print("DONE")

#extract elements from lists:

print("---------------------------------------------------------------------------------")
print("Extracting layers and momenta...")

#momentum
p <- sapply(dat,`[[`,"P")
#pdfg code
pdg <- sapply(dat, `[[`,"pdgCode")
#layers
layer_0 <- sapply(dat, `[[`,"layer 0")
layer_1 <- sapply(dat, `[[`,"layer 1")
layer_2 <- sapply(dat, `[[`,"layer 2")
layer_3 <- sapply(dat, `[[`,"layer 3")
layer_4 <- sapply(dat, `[[`,"layer 4")
layer_5 <- sapply(dat, `[[`,"layer 5")

print("---------------------------------------------------------------------------------")
print("Removing NULLS")

#remove tracklets with less than 6 layers
n <- unique(
  c(
    which(sapply(layer_0, is.null)),
    which(sapply(layer_1, is.null)),
    which(sapply(layer_2, is.null)),
    which(sapply(layer_3, is.null)),
    which(sapply(layer_4, is.null)),
    which(sapply(layer_5, is.null))
  ),nmax=100000000000000
)

pdg <- pdg[-n]

layer_0 <- layer_0[-n]

layer_1 <- layer_1[-n]

layer_2 <- layer_2[-n]

layer_3 <- layer_3[-n]

layer_4 <- layer_4[-n]

layer_5 <- layer_5[-n]

p <- p[-n]

print("---------------------------------------------------------------------------------")
print("Removing empties:")

#remove tracklets that passed through detector elements that didn't return data

e <- unique(
  
  as.numeric(which(sapply(layer_0, "typeof")=="list")),
  as.numeric(which(sapply(layer_1, "typeof")=="list")),
  as.numeric(which(sapply(layer_2, "typeof")=="list")),
  as.numeric(which(sapply(layer_3, "typeof")=="list")),
  as.numeric(which(sapply(layer_4, "typeof")=="list")),
  as.numeric(which(sapply(layer_5, "typeof")=="list")),nmax=100000000000000
  
)

pdg <- pdg[-e]

layer_0 <- layer_0[-e]

layer_1 <- layer_1[-e]

layer_2 <- layer_2[-e]

layer_3 <- layer_3[-e]

layer_4 <- layer_4[-e]

layer_5 <- layer_5[-e]

p <- p[-e]

#check if all tracklets have 6 layers, and initialize an empty x
#array of 17 pads, 24 timebins and 6 layers

if(length(layer_0)==length(layer_1) &&
   length(layer_1) ==length(layer_2) &&
   length(layer_2)==length(layer_3)&&
   length(layer_3)==length(layer_4)&&
   length(layer_4)==length(layer_5)){
  
  print("Layer dimensions check out")
  x <- array(dim=c(1,17,24,6))
  
}else{
  #layers missing
  print("Layers are not same dimensions, stopping")
  stop()
}

#append said tensor for each tracklet

print("Appending tensors")

require(abind,lib="/scratch/vljchr004/")

for(i in 1:500){#length(layer_0)){

print(100*i/500)#length(layer_0))

  a <- array(data=c(
    
    layer_0[[i]], 
    layer_1[[i]], 
    layer_2[[i]], 
    layer_3[[i]], 
    layer_4[[i]], 
    layer_5[[i]]
    
  ),
  dim = c(17,24,6)
  )
  
  x <- abind(x,a,along = 1)

  x[i,,,]
}

#remove empty element used to initialize x array

x <- x[-1,,,]

#check if dimensions of array check out and if so load the keras package

if(dim(x)[2]==17 &&
   dim(x)[3]==24 &&
   dim(x)[4]==6 &&
   dim(x)[1]==500){#length(layer_0)){
  #require(keras, lib.loc = "/scratch/vljchr004/")
print("initiate fuck-up")
}else{
  print("Array building seems to not have went well, stopping")
  stop()
  
}

require(keras, lib.loc = "/scratch/vljchr004/")
#system("source activate r-tensorflow")

#use_condaenv("r-tensorflow")

#require(keras, lib.loc = "/scratch/vljchr004/")

#use_condaenv("r-tensorflow")


print("Y to categorical")

y <- as.vector(ifelse(abs(as.numeric(pdg))==211,1,0))
y <- to_categorical(y)


print("defining keras CNN")

cnn_model <- keras_model_sequential()

cnn_model %>%
  
  #layer conv 1
  
  layer_conv_2d(filter=32,
                kernel_size=c(3,3),
                padding="same",
                input_shape=c(17,24,6) ) %>%  
  
  layer_activation("relu") %>%  
  
  #layer conv 2
  
  layer_conv_2d(filter=32 ,
                kernel_size=c(3,3)) %>%
  
  layer_activation("relu") %>%
  
  #max pooling layer

  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  
  #dropout layer to avoid overfitting
  
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter=32 ,
                kernel_size=c(3,3),
                padding="same") %>%
  
  layer_activation("relu") %>%
  
  layer_conv_2d(filter=32,kernel_size=c(3,3) ) %>%
  
  layer_activation("relu") %>%  
  
  
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  
  #flatten the input  
  layer_flatten() %>%  
  
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  
  layer_dropout(0.5) %>%  
  
  #output layer-10 classes-10 units  
  layer_dense(10) %>%  
  
  #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy  
  
  layer_activation("softmax") 

#for computing Probabilities of classes-"logit(log probabilities)

#Model's Optimizer

#defining the type of optimizer-ADAM-Adaptive Momentum Estimation

opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )

#lr-learning rate , decay - learning rate decay over each update

print("Running keras model")

cnn_model %>%
  compile(loss="categorical_crossentropy",
          optimizer=opt,metrics = "accuracy")

#Summary of the Model and its Architecture
summary(cnn_model)


#training and test:

set.seed(123456)

train_ind <- base::sample(1:dim(x)[4],size = round(0.75*dim(x)[4]),replace = F)

x_train <- x[train_ind,,,]
y_train <- y[train_ind]

test_ind <- 1:dim(x)[4]
test_ind <- test_ind[-train_ind]

x_test <- x[test_ind,,,]
y_test <- y[test_ind]

save(x_train,file="/scratch/vljchr004/x_train")
save(y_train,file="/scratch/vljchr004/y_train")
save(x_test,file="/scratch/vljchr004/x_test")
save(y_train,file="/scratch/vljchr004/y_train")
history <- cnn_model %>%
  fit(x_train,
      y_train,
      batch_size=64,
      epochs=100,
      validation_split=0.3)

png("/home/vljchr004/cnn_model_1.png")

plot(history)

dev.off()

cnn_model %>% save_model_hdf5("/scratch/vljchr004/my_model.h5")

p <- predict_proba(cnn_model,x_test)

p <- cbind(p,y_test)

save(p,file="/scratch/vljchr004/p.rdata")

write.csv(p,file="/scratch/vljchr004/p.csv")
