library(geigen)

kernelG <- function(A, B, sigma) {
  na <- nrow(A)
  nb <- nrow(B)

  K <- matrix(0, nrow = na, ncol = nb)

  for (i in 1:na) {
    for (j in 1:nb) {

      x <- -norm(A[i, ] - B[j, ], type = "2")^2
      K[i, j] <- exp(x/sigma)
    }
  }
  return(K)
}

train_regec <- function(train_data, train_labels, sigma = 900, delta1 = 0.1, delta2 = 0.1) {

  A <- train_data[train_labels == 0, ]
  B <- train_data[train_labels == 1, ]
  C <- rbind(A, B)


  # Building left and right matrices
  g <- cbind(kernelG(A, C, sigma), -matrix(1, nrow = nrow(A)))
  h <- cbind(kernelG(B, C, sigma), -matrix(1, nrow = nrow(B)))
  G1 <- t(g) %*% g
  H1 <- t(h) %*% h
 
  T <- diag(diag(H1))
  U <- diag(diag(G1))

  # Build planes
  G <- G1 + delta1 * T
  H <- H1 + delta2 * U

  eig <- geigen(G, H)
  w <- eig$values
  vr <- eig$vectors
  imin1 <- which.min(w)
  imax2 <- which.max(w)
  W <- matrix(vr[, c(imin1, imax2)], ncol = 2)

  n <- nrow(C)

  trained_model <- list(sigma = sigma, delta1 = delta1, delta2 = delta2, C = C, vr = vr, n = n, imin1 = imin1, imax2 = imax2, W = W)

  return(trained_model)
}

predict_regec <- function(trained_model, test_data) {
  sigma <- trained_model$sigma
  C <- trained_model$C
  vr <- trained_model$vr
  n <- trained_model$n
  imin1 <- trained_model$imin1
  imax2 <- trained_model$imax2
  W <- trained_model$W

  K <- kernelG(test_data, C, sigma)
  z1 <- abs(K %*% vr[0:n, imin1] - vr[n, imin1])^2 / norm(as.matrix(vr[0:n, imin1]))^2
  z2 <- abs(K %*% vr[0:n, imax2] - vr[n, imax2])^2 / norm(as.matrix(vr[0:n, imax2]))^2

  class_l <- sign(- z1 + z2)
  
  return(list(class_l = class_l))
}

# Load data
train_file <- "train.csv"
test_file <- "test.csv"
train <- read.csv(train_file, header = FALSE)
test <- read.csv(test_file, header = FALSE)

# Check for missing values and remove them
train_missing <- sum(is.na(train))
test_missing <- sum(is.na(test))
if (train_missing > 0) {
  train <- train[complete.cases(train), ]
}
if (test_missing > 0) {
  test <- test[complete.cases(test), ]
}

# Extract labels and features
train_l <- train[, ncol(train)]
test_l <- test[, ncol(test)]
train <- train[, -ncol(train)]
test <- test[, -ncol(test)]

# Train model
trained_model <- train_regec(train, train_l, sigma=900, delta1=0.1, delta2=0.1)

# Predict using trained model
predictions <- predict_regec(trained_model, test)

# Print accuracy
cat("Accuracy:", mean(predictions$class_l == test_l))
