rm(list=ls())

require(jsonlite)
require(readtext)

files <- list.files(path="/Users/gerhard/msc-thesis-data/000265309/JS",
                    pattern="*json",
                    full.names=T,
                    recursive=FALSE)

j <- fromJSON(files[2])

for(i in 3:length(files)){
  
  f <- fromJSON(files[i])
  j <- c(j,f)
}

length(j)

save(j,file="/Users/gerhard/msc-thesis-data/000265309/RDATA/000265309_fulljson.rdata")