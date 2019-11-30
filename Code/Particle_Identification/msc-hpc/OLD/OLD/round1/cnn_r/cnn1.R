args = commandArgs(TRUE)
x_train_f = args[1]
y_train_f = args[2]
x_test_f = args[3]
y_test_f = args[4]
x_valid_f = args[5]
y_valid_f = args[6]

train_x <- load(x_train_f)
train_y <- load(y_train_f)
test_x <- load(x_test_f)
test_y <- load(y_test_f)
valid_x <- load(x_valid_f)
valid_y <- load(y_valid_f)

model <- keras_model_sequential()

model %>%
  

  layer_conv_1d(
    filter = 24, kernel_size = 3, padding = "causal", 
    input_shape = c(6,24)
  ) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_1d(pool_size = c(2)) %>%
  layer_dropout(0.25) %>%
  
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
  validation_data = list(valid_x, valid_y),
  shuffle = TRUE
)

png("/scratch/vljchr004/first_cnn.png")
plot(history)
dev.off()

p <- predict_proba(model,test_x)


save_model_hdf5(model,filepath="/scratch/vljchr004/cnn1.h5")

write.csv(p,file="/scratch/vljchr004/cnn1_test_preds.csv")
write.csv(test_y,"/scratch/vljchr004/cnn1_test_y.csv")


print("done")
