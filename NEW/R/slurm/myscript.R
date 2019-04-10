args = commandArgs(TRUE)
input1 = args[1]
input2 = args[2]

if(!file.exists(input1)|!file.exists(input2)){
   cat('Cannot find', input_path, 'exiting!\n')
   stop()
}

load(input1)
load(input2)

head(x)

#install.packages(pkgs="keras",lib="/scratch/vljchr004",
#repos="https://cloud.r-project.org",dependencies=TRUE)

#system("source activate /home/vljchr004/.conda/envs/r-tensorflow")

require(keras, lib.loc = "/scratch/vljchr004/")


#install_keras()

#keras::install_tensorflow()

y <- to_categorical(y)

print(head(y))
print("eureka")

