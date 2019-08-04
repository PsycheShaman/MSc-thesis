rm(list=ls())

sim_files <- list.files(path="C:/Users/gerhard/Documents/msc-thesis-data/hijing-sim/", pattern="*.json", full.names=T, recursive=T)

require(jsonlite)

j <- list()

count=0
for(i in sim_files){
  j <- c(j,fromJSON(i))
}


##

layer0 <- sapply(j, `[[`,"layer 0")

nSigmaPion0 <- sapply(j, `[[`,"nSigmaPion")

P0 <- sapply(j, `[[`,"P")

Eta0 <- sapply(j, `[[`,"Eta")

n <- unique(c(which(sapply(layer0,typeof)=="list"),which(sapply(layer0, is.null))))

length(n)

layer0 <- layer0[-n]
nSigmaPion0 <- nSigmaPion0[-n]
P0 <- P0[-n]
Eta0 <- Eta0[-n]

for(i in 1:length(layer0)){
  layer0[[i]][which(layer0[[i]]==-7169)] <- 0
}

##

layer1 <- sapply(j, `[[`,"layer 1")

nSigmaPion1 <- sapply(j, `[[`,"nSigmaPion")

P1 <- sapply(j, `[[`,"P")

Eta1 <- sapply(j, `[[`,"Eta")


n <- unique(c(which(sapply(layer1,typeof)=="list"),which(sapply(layer1, is.null))))

length(n)

layer1 <- layer1[-n]
nSigmaPion1 <- nSigmaPion1[-n]
P1 <- P1[-n]
Eta1 <- Eta1[-n]

for(i in 1:length(layer1)){
  layer1[[i]][which(layer1[[i]]==-7169)] <- 0
}

##

layer2 <- sapply(j, `[[`,"layer 2")

nSigmaPion2 <- sapply(j, `[[`,"nSigmaPion")

P2 <- sapply(j, `[[`,"P")

Eta2 <- sapply(j, `[[`,"Eta")

n <- unique(c(which(sapply(layer2,typeof)=="list"),which(sapply(layer2, is.null))))

length(n)

layer2 <- layer2[-n]
nSigmaPion2 <- nSigmaPion2[-n]
P2 <- P2[-n]
Eta2 <- Eta2[-n]

for(i in 1:length(layer2)){
  layer2[[i]][which(layer2[[i]]==-7169)] <- 0
}

##

layer3 <- sapply(j, `[[`,"layer 3")

nSigmaPion3 <- sapply(j, `[[`,"nSigmaPion")

P3 <- sapply(j, `[[`,"P")
Eta3 <- sapply(j, `[[`,"Eta")


n <- unique(c(which(sapply(layer3,typeof)=="list"),which(sapply(layer3, is.null))))

length(n)

layer3 <- layer3[-n]
nSigmaPion3 <- nSigmaPion3[-n]
P3 <- P3[-n]
Eta3 <- Eta3[-n]

for(i in 1:length(layer3)){
  layer3[[i]][which(layer3[[i]]==-7169)] <- 0
}

##

layer4 <- sapply(j, `[[`,"layer 4")

nSigmaPion4 <- sapply(j, `[[`,"nSigmaPion")

P4 <- sapply(j, `[[`,"P")

Eta4 <- sapply(j, `[[`,"Eta")

n <- unique(c(which(sapply(layer4,typeof)=="list"),which(sapply(layer4, is.null))))

length(n)

layer4 <- layer4[-n]
nSigmaPion4 <- nSigmaPion4[-n]
P4 <- P4[-n]
Eta4 <- Eta4[-n]

for(i in 1:length(layer4)){
  layer4[[i]][which(layer4[[i]]==-7169)] <- 0
}

##

layer5 <- sapply(j, `[[`,"layer 5")

nSigmaPion5 <- sapply(j, `[[`,"nSigmaPion")

P5 <- sapply(j, `[[`,"P")

Eta5 <- sapply(j, `[[`,"Eta")

n <- unique(c(which(sapply(layer5,typeof)=="list"),which(sapply(layer5, is.null))))

length(n)

layer5 <- layer5[-n]
nSigmaPion5 <- nSigmaPion5[-n]
P5 <- P5[-n]
Eta5 <- Eta5[-n]

for(i in 1:length(layer5)){
  layer5[[i]][which(layer5[[i]]==-7169)] <- 0
}

##

rm(j)

sim_x <- c(layer0,layer1,layer2,layer3,layer4,layer5)

rm(layer0,layer1,layer2,layer3,layer4,layer5)

for(i in 1:length(sim_x)){
  dim(sim_x[[i]]) <- c(1,17,24)
  # print(i/length(sim_x))
  
}

require(abind)

sim_x <- abind(sim_x,along=1)



nsigmaPion <- c(nSigmaPion0,nSigmaPion1,nSigmaPion2,nSigmaPion3,nSigmaPion4,nSigmaPion5)
nsigmaPion <- as.numeric(unlist(nsigmaPion))

length(nsigmaPion)==dim(sim_x)[1]

rm(nSigmaPion0,nSigmaPion1,nSigmaPion2,nSigmaPion3,nSigmaPion4,nSigmaPion5)

P <- c(P0,P1,P2,P3,P4,P5)
P <- as.numeric(unlist(P))

length(P)==dim(sim_x)[1]

rm(P0,P1,P2,P3,P4,P5)

sim_P <- P

rm(P)

Eta <- c(Eta0,Eta1,Eta2,Eta3,Eta4,Eta5)
Eta <- as.numeric(unlist(Eta))

length(Eta)==dim(sim_x)[1]

rm(Eta0,Eta1,Eta2,Eta3,Eta4,Eta5)

sim_Eta <- Eta

rm(Eta)

sim_nsig <- nsigmaPion
rm(nsigmaPion)

rm(list=ls())

real_files <- list.files(path="C:/Users/gerhard/Documents/msc-thesis-data/processed/000265343/", pattern="*.json", full.names=T, recursive=T)

require(jsonlite)

j <- list()
count <- 0
for(i in real_files){
  j <- c(j,fromJSON(i))
}


##

layer0 <- sapply(j, `[[`,"layer 0")

nSigmaPion0 <- sapply(j, `[[`,"nSigmaPion")
pdgCode0 <- sapply(j, `[[`,"pdgCode")

P0 <- sapply(j, `[[`,"P")

Eta0 <- sapply(j, `[[`,"Eta")



n <- unique(c(which(sapply(layer0,typeof)=="list"),which(sapply(layer0, is.null))))

length(n)

layer0 <- layer0[-n]
nSigmaPion0 <- nSigmaPion0[-n]
pdgCode0 <- pdgCode0[-n]
P0 <- P0[-n]
Eta0 <- Eta0[-n]

##

layer1 <- sapply(j, `[[`,"layer 1")

nSigmaPion1 <- sapply(j, `[[`,"nSigmaPion")
pdgCode1 <- sapply(j, `[[`,"pdgCode")
P1 <- sapply(j, `[[`,"P")
Eta1 <- sapply(j, `[[`,"Eta")
n <- unique(c(which(sapply(layer1,typeof)=="list"),which(sapply(layer1, is.null))))

length(n)

layer1 <- layer1[-n]
nSigmaPion1 <- nSigmaPion1[-n]
pdgCode1 <- pdgCode1[-n]
P1 <- P1[-n]
Eta1 <- Eta1[-n]

##

layer2 <- sapply(j, `[[`,"layer 2")

nSigmaPion2 <- sapply(j, `[[`,"nSigmaPion")
pdgCode2 <- sapply(j, `[[`,"pdgCode")
P2 <- sapply(j, `[[`,"P")
Eta2 <- sapply(j, `[[`,"Eta")
n <- unique(c(which(sapply(layer2,typeof)=="list"),which(sapply(layer2, is.null))))

length(n)

layer2 <- layer2[-n]
pdgCode2 <- pdgCode2[-n]
nSigmaPion2 <- nSigmaPion2[-n]
P2 <- P2[-n]
Eta2 <- Eta2[-n]

##

layer3 <- sapply(j, `[[`,"layer 3")

nSigmaPion3 <- sapply(j, `[[`,"nSigmaPion")
pdgCode3 <- sapply(j, `[[`,"pdgCode")
P3 <- sapply(j, `[[`,"P")
Eta3 <- sapply(j, `[[`,"Eta")
n <- unique(c(which(sapply(layer3,typeof)=="list"),which(sapply(layer3, is.null))))

length(n)

layer3 <- layer3[-n]
pdgCode3 <- pdgCode3[-n]
nSigmaPion3 <- nSigmaPion3[-n]
P3 <- P3[-n]
Eta3 <- Eta3[-n]

##

layer4 <- sapply(j, `[[`,"layer 4")

nSigmaPion4 <- sapply(j, `[[`,"nSigmaPion")
pdgCode4 <- sapply(j, `[[`,"pdgCode")
P4 <- sapply(j, `[[`,"P")
Eta4 <- sapply(j, `[[`,"Eta")
n <- unique(c(which(sapply(layer4,typeof)=="list"),which(sapply(layer4, is.null))))

length(n)

layer4 <- layer4[-n]
pdgCode4 <- pdgCode4[-n]
nSigmaPion4 <- nSigmaPion4[-n]
P4 <- P4[-n]
Eta4 <- Eta4[-n]

##

layer5 <- sapply(j, `[[`,"layer 5")

nSigmaPion5 <- sapply(j, `[[`,"nSigmaPion")
pdgCode5 <- sapply(j, `[[`,"pdgCode")
P5 <- sapply(j, `[[`,"P")
Eta5 <- sapply(j, `[[`,"Eta")
n <- unique(c(which(sapply(layer5,typeof)=="list"),which(sapply(layer5, is.null))))

length(n)

layer5 <- layer5[-n]
pdgCode5 <- pdgCode5[-n]
nSigmaPion5 <- nSigmaPion5[-n]
P5 <- P5[-n]
Eta5 <- Eta5[-n]

##

rm(j)

real_x <- c(layer0,layer1,layer2,layer3,layer4,layer5)

rm(layer0,layer1,layer2,layer3,layer4,layer5)

for(i in 1:length(real_x)){
  dim(real_x[[i]]) <- c(1,17,24)
  
}

require(abind)

real_x <- abind(real_x,along=1)



nsigmaPion <- c(nSigmaPion0,nSigmaPion1,nSigmaPion2,nSigmaPion3,nSigmaPion4,nSigmaPion5)
nsigmaPion <- as.numeric(unlist(nsigmaPion))

pdgCode <- c(pdgCode0,pdgCode1,pdgCode2,pdgCode3,pdgCode4,pdgCode5)
pdgCode <- as.numeric(unlist(pdgCode))

rm(nSigmaPion0,nSigmaPion1,nSigmaPion2,nSigmaPion3,nSigmaPion4,nSigmaPion5)
rm(pdgCode0,pdgCode1,pdgCode2,pdgCode3,pdgCode4,pdgCode5)

P <- c(P0,P1,P2,P3,P4,P5)
P <- as.numeric(unlist(P))

rm(P0,P1,P2,P3,P4,P5)

real_P <- P

rm(P)

Eta <- c(Eta0,Eta1,Eta2,Eta3,Eta4,Eta5)
Eta <- as.numeric(unlist(Eta))

rm(Eta0,Eta1,Eta2,Eta3,Eta4,Eta5)

real_Eta <- Eta

rm(Eta)

pions <- which(abs(pdgCode)==211)

nsigmaPion <- nsigmaPion[pions]
real_x <- real_x[pions,,]
real_Eta <- real_Eta[pions]
real_P <- real_P[pions]

length(nsigmaPion)==dim(real_x)[1]

real_nsig <- nsigmaPion

rm(nsigmaPion)

Eta <- data.frame(rbind(cbind(real_Eta,"real"),cbind(sim_Eta,"sim")))

names(Eta) <- c("Eta","real_or_fake")

Eta$Eta <- as.numeric(as.character(Eta$Eta))
Eta$real_or_fake <- as.factor(Eta$real_or_fake)

P <- data.frame(rbind(cbind(real_P,"real"),cbind(sim_P,"sim")))

names(P) <- c("P","real_or_fake")

P$P <- as.numeric(as.character(P$P))
P$real_or_fake <- as.factor(P$real_or_fake)

P <- data.frame(P)

nsig <- data.frame(rbind(cbind(real_nsig,"real"),cbind(sim_nsig,"sim")))

names(nsig) <- c("nsig","real_or_fake")

nsig$nsig <- as.numeric(as.character(nsig$nsig))
nsig$real_or_fake <- as.factor(nsig$real_or_fake)


require(abind)
x <- abind(real_x,sim_x,along=1)
y <- c(rep(1,dim(real_x)[1]),rep(0,dim(sim_x)[1]))

x <- x[abs(Eta$Eta)<=.9,,]

nsig <- nsig[(abs(Eta$Eta))<=.9,]

P <- P[(abs(Eta$Eta))<=.9,]

y <- y[(abs(Eta$Eta))<=.9]

x <- x[P$P<=20,,]

nsig <- nsig[P$P<=20,]

y <- y[P$P<=20]

x <- x[abs(nsig$nsig)<=3,,]

y <- y[abs(nsig$nsig)<=3]


rm(real_Eta,real_nsig,real_P,sim_Eta,sim_nsig,sim_P,real_x,sim_x,Eta,nsig,P,P1)

rm(real_x,sim_x)

require(keras)

dim(x) <- c(dim(x),1)

x <- (x-max(x))/max(x)

train_ind <- sample(1:length(y),size=round(0.75*length(y)))

x_train <- x[train_ind,,,]
y_train <- y[train_ind]

x_test <- x[-train_ind,,,]
y_test <- y[-train_ind]

dim(x_train) <- c(dim(x_train),1)
dim(x_test) <- c(dim(x_test),1)


real <- which(y==1)
fake <- which(y!=1)

i <- c(fake,sample(real,length(fake),replace=F))

x <- x[i,,,]
y <- y[i]
dim(x) <- c(dim(x),1)


train_ind <- sample(1:length(y),size=round(0.75*length(y)))

x_train <- x[train_ind,,,]
y_train <- y[train_ind]

x_test <- x[-train_ind,,,]
y_test <- y[-train_ind]

final_test_ind <- sample(1:length(y_train),size=round(0.1*length(y_train)))

x_train <- x[-final_test_ind,,,]
y_train <- y[-final_test_ind]

x_final_test <- x[final_test_ind,,,]
y_final_test <- y[final_test_ind]


dim(x_train) <- c(dim(x_train),1)
dim(x_test) <- c(dim(x_test),1)
dim(x_final_test) <- c(dim(x_final_test),1)

rm(x,y)

model <- keras_model_sequential() %>%
  layer_conv_2d(16, kernel_size = c(2,3)) %>%
  layer_max_pooling_2d() %>%
  layer_conv_2d(32, kernel_size = c(3,3)) %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>%
  layer_dense(128,"tanh") %>%
  layer_dense(1, "sigmoid")

model %>%
  compile(
    loss="binary_crossentropy",
    optimizer_adam(0.001),
    metrics="acc"
    
  )

history <- model %>%
  fit(x_train,
      y_train,
      batch_size=8,
      verbose=2,
      epochs=20,
      validation_data=list(x_test,y_test),
      shuffle=TRUE)

png("C:/Users/gerhard/documents/MSc-thesis/NEW/ML/geant_v_real5.png")
plot(history)
dev.off()









































