library(geigen)
library(caret)
library(ggplot2)

# Parameters
sigma <- 900
delta1 <- 0.1
delta2 <- 0.1

gKernel <- function(A, B, sigma) {
  # Compute the pairwise Euclidean distances between rows of A and B
  dists <- as.matrix(dist(rbind(A, B)))
  distsX <- dists[1:nrow(A), 1:nrow(B)]
  
  # Compute the Gaussian kernel matrix
  K <- exp(-distsX^2 / (2 * sigma^2))
  
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
  left <- cbind(gKernel(A, C, sigma), rep(-1, nrow(A)))
  right <- cbind(gKernel(B, C, sigma), rep(-1, nrow(B)))

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

# split_dataset <- function(file_path) {
#   # Read in the dataset from file
#   dataset <- read.csv(file_path)
#
#   # Set seed for reproducibility
#   set.seed(123)
#
#   # Get the number of rows in the dataset
#   n_rows <- nrow(dataset)
#
#   # Calculate the number of rows for the training set and the testing set
#   train_size <- round(0.6 * n_rows)
#   test_size <- n_rows - train_size
#
#   # Randomly select rows for the training set
#   train_index <- sample(seq_len(n_rows), size = train_size, replace = FALSE)
#   train_set <- dataset[train_index, ]
#
#   # Select the remaining rows for the testing set
#   test_set <- dataset[-train_index, ]
#
#   # Return the training and testing sets as a list
#   #return(list(train = train_set, test = test_set))
#
#   # Write the training and testing sets to separate CSV files
#   write.csv(train_set, file = "train.csv", row.names = FALSE)
#   write.csv(test_set, file = "test.csv", row.names = FALSE)
# }
#
# dataset_split <- split_dataset("regecR/heart_cleveland_upload.csv")

# Load diabetes.csv dataset
dataset <- read.csv("regecR/heart_cleveland_upload.csv")

# Split dataset into training and testing sets
set.seed(123) # set random seed for reproducibility
train_indices <- createDataPartition(dataset$condition, p = 0.6, list = FALSE)
train_data <- dataset[train_indices, ]
test_data <- dataset[-train_indices, ]

# Save training and testing sets as CSV files
write.csv(train_data, file = "train.csv", row.names = FALSE)
write.csv(test_data, file = "test.csv", row.names = FALSE)

# Load data
train_file <- "train.csv"
test_file <- "test.csv"
train <- read.csv(train_file)
test <- read.csv(test_file)

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
