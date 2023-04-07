library(geigen)
library(httpgd)
library(ggplot2)
library(gridExtra)

hgd()
hgd_browse()

# Parameters

sigma <- 900
delta1 <- 0.1
delta2 <- 0.1

# Compute the Gaussian kernel matrix between A and B
gKernel <- function(A, B, sigma) {
  na <- nrow(A)
  nb <- nrow(B)

  K <- matrix(0, nrow = na, ncol = nb)

  for (i in 1:na) {
    for (j in 1:nb) {

      dist <- -norm(A[i, ] - B[j, ], type = "2")^2
      K[i, j] <- exp(dist/sigma)
    }
  }
  return(K)
}

# Train a regularized kernel-based classifier on a labeled dataset
train_regec <- function(train_data, train_labels, sigma = sigma, delta1 = delta1, delta2 = delta2) {

  # Separate the dataset into two subsets, one for each class
  A <- train_data[train_labels == 0, ]
  B <- train_data[train_labels == 1, ]

  # Combine the two subsets to form a larger dataset C
  C <- rbind(A, B)


  # Build the left and right matrices
  left <- cbind(gKernel(A, C, sigma), -matrix(1, nrow = nrow(A)))
  right <- cbind(gKernel(B, C, sigma), -matrix(1, nrow = nrow(B)))

  # Compute T and U, which are diagonal matrices used to regularize the classifier
  T <- diag(diag(t(right) %*% right))
  U <- diag(diag(t(left) %*% left))

  # Build the planes used to separate the two classes
  G <- t(left) %*% left + delta1 * T
  H <- t(right) %*% right + delta2 * U

  # Compute the generalized eigenvectors and eigenvalues of G and H
  eig <- geigen(G, H)
  eigenvalues <- eig$values
  eigenvectors <- eig$vectors
  imin1 <- which.min(eigenvalues)
  imax2 <- which.max(eigenvalues)
  W <- matrix(eigenvectors[, c(imin1, imax2)], ncol = 2)

  n <- nrow(C)

  # Store the trained model parameters in a list
  trained_model <- list(sigma = sigma, delta1 = delta1, delta2 = delta2, C = C, v = eigenvectors, n = n, imin1 = imin1, imax2 = imax2, W = W)

  return(trained_model)
}

# Use a trained regularized kernel-based classifier to make predictions on a new dataset
predict_regec <- function(trained_model, test_data) {
  sigma <- trained_model$sigma
  C <- trained_model$C
  v <- trained_model$v
  n <- trained_model$n
  imin1 <- trained_model$imin1
  imax2 <- trained_model$imax2

  # Compute the Gaussian kernel matrix between the test dataset and the training dataset
  K <- gKernel(test_data, C, sigma)

  # Compute the distances between the test dataset and the two separating planes
  z1 <- abs(K %*% v[0:n, imin1] - v[n, imin1])^2 / norm(as.matrix(v[0:n, imin1]))^2
  z2 <- abs(K %*% v[0:n, imax2] - v[n, imax2])^2 / norm(as.matrix(v[0:n, imax2]))^2

  class_labels <- sign(- z1 + z2)
  
  return(list(class_labels = class_labels))
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
train_labels <- train[, ncol(train)]
test_labels <- test[, ncol(test)]
train <- train[, -ncol(train)]
test <- test[, -ncol(test)]

# Train model
trained_model <- train_regec(train, train_labels, sigma = sigma, delta1 = delta1, delta2 = delta2)

# Predict using trained model
predictions <- predict_regec(trained_model, test)

# Print accuracy
cat("Accuracy:", mean(predictions$class_labels == test_labels))


# Plot predicted class against true class
p1 <- ggplot(data.frame(y_true = test_labels, y_pred = predictions$class_labels), aes(x = y_true, y = y_pred)) + 
      geom_point() + geom_abline(intercept = 0, slope = 1, color = "red") + 
      ggtitle("Predicted vs True Class") + xlab("True Class") + ylab("Predicted Class")

plot(p1)

# Plot decision boundary
x1 <- seq(min(test[, 1]), max(test[, 1]), length.out = 100)
x2 <- seq(min(test[, 2]), max(test[, 2]), length.out = 100)
grid <- expand.grid(x1 = x1, x2 = x2)
X <- as.matrix(grid)
Z <- predict_regec(trained_model, X)$class_labels
df <- data.frame(x1 = X[, 1], x2 = X[, 2], z = Z)
p2 <- ggplot(df, aes(x = x1, y = x2, fill = factor(z))) + 
      geom_tile() + 
      ggtitle("Decision Boundary") + xlab("Feature 1") + ylab("Feature 2") + 
      scale_fill_manual(values = c("red", "green"))

plot(p2)

# Combine plots
p3 <- grid.arrange(p1, p2, nrow = 1, widths = c(0.6, 0.4))

plot(p3)
