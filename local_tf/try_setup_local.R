install.packages("keras")

require(keras)

#install_keras(tensorflow = "gpu")

require(reticulate)

reticulate::use_condaenv("py35",required = T)

install_keras(tensorflow="gpu")

file.edit('~/.Renviron')

require(keras)
