rm(list=ls())

require(jsonlite)
require(readtext)

files <- list.files(path="/Users/gerhard/Thesis-data/JS/000265377",
                    pattern="*json",
                    full.names=T,
                    recursive=FALSE)

j <- fromJSON(files[2])

for(i in 3:length(files)){
  
  f <- fromJSON(files[i])
  j <- c(j,f)
}

length(j)

save(j,file="/Users/gerhard/Thesis-data/RDATA/000265377/000265377_fulljson.rdata")