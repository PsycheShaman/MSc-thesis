args = commandArgs(TRUE)

x_file = args[1]
y_file = args[2]

load(x_file)
load(y_file)

(!file.exists(x_file)){
  cat('Cannot find', x_file, 'exiting!\n')
  stop()
}

(!file.exists(x_file)){
  cat('Cannot find', y_file, 'exiting!\n')
  stop()
}

install.packages(pkgs="keras",repos = "https://cloud.r-project.org",lib = "~",dependencies=TRUE)

require(keras)
install_keras()















