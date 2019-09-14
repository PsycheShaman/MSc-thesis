#rm(list=ls())

wrangle <- function(run){
  
  require(jsonlite)
  require(readtext)
  
  files <- list.files(path=paste0("/Users/gerhard/msc-thesis-data/000",run,"/JS"),
                      pattern="*json",
                      full.names=T,
                      recursive=TRUE)
  
  j <- fromJSON(files[2])
  
  for(i in 3:length(files)){
    
    f <- fromJSON(files[i])
    j <- c(j,f)
  }
  
j
  
}

dat1 <- wrangle("265309")
dat2 <- wrangle("265377")
dat3 <- wrangle("265378")

