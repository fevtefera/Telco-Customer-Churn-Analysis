install.packages("caret")
library(caret)
#load and explore dataset
telco_data <- read.csv("Customer-Churn.csv")
str(telco_data)
summary(telco_data)
#remove customer ID because it has no useful information
churn_data <-telco_data[,-1]
str(churn_data)
numeric_predictors <- churn_data[sapply(churn_data, is.numeric)]
num_cols <- 3
num_rows <- ceiling(length(colnames(numeric_predictors)) / num_cols)
par(mfrow = c(num_rows, num_cols))
#Plot histograms for each numeric predictor
for (predictor in colnames(numeric_predictors)) {
  hist(
    numeric_predictors[[predictor]],
    main = paste("Histogram of", predictor),
    xlab = predictor,
    col = "skyblue",
    border = "black",
    breaks = 20
  )
}
par(mfrow = c(1, 1))
# missing values
#colSums(is.na(churn_data))
# checking the column with missing value
missing_total_charges <- sum(is.na(churn_data$TotalCharges))
missing_total_charges
#Since we have only 11 missing values we remove the 11 rows.
data <- na.omit(churn_data)
#checking the missing values after removing the rows with missing values
missing_total_charges <- sum(is.na(data))
missing_total_charges
#checking the dimension of the data after removing the rows with null values
dim(data)
str(data)
#check response variable balance
barplot(table(data$Churn),
        main = "Distribution of Customer Churn",
        xlab = "Churn",
        ylab = "Frequency",
        col = c("lightblue", "salmon"),
        border = "black")
table(data$Churn)
#Separate predictors and target variable
predictors <- data[, -which(names(data) == "Churn")]
target <- data$Churn
sapply(predictors, function(x) if (is.character(x) | is.factor(x)) unique(x))
# Binary encoding for categorical variables with two levels
predictors$gender <- ifelse(predictors$gender == "Female", 1, 0)
predictors$Partner <- ifelse(predictors$Partner == "Yes", 1, 0)
predictors$Dependents <- ifelse(predictors$Dependents == "Yes", 1, 0)
predictors$PhoneService <- ifelse(predictors$PhoneService == "Yes", 1, 0)
predictors$PaperlessBilling <- ifelse(predictors$PaperlessBilling == "Yes", 1, 0)
# List of multi-level categorical variables
multi_level_vars <- c("MultipleLines", "InternetService", "OnlineSecurity",
                      "OnlineBackup", "DeviceProtection", "TechSupport",
                      "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod")
# Select only multi-level variables from predictors
multi_level_data <- predictors[, multi_level_vars]
# Create dummy variables for the multi-level variables
dummy_model <- dummyVars(~ ., data = multi_level_data, fullRank = TRUE)
multi_level_encoded <- data.frame(predict(dummy_model, newdata = multi_level_data))
multi_level_encoded
# Remove the original multi-level columns from predictors
predictors <- predictors[, !names(predictors) %in% multi_level_vars]
# Add the dummy-encoded multi-level variables back to predictors
predictors <- cbind(predictors, multi_level_encoded)
str(predictors)
library(caret)
# Analyze near-zero variance
nzv <- nearZeroVar(predictors, saveMetrics = TRUE)
nzv_count <- sum(nzv$nzv)
nzv_count
install.packages("corrplot")
library(corrplot)
# Extract numeric predictors
num_predictors <- predictors[sapply(predictors, is.numeric)]
correlation <- cor(num_predictors, use = "complete.obs")
tooHigh <- findCorrelation(correlation, cutoff = 0.75)
num_high_corr <- length(tooHigh)
cat("Number of Highly Correlated Predictors:", num_high_corr, "\n")
# Plot the correlation matrix
corrplot(correlation, method = "circle", order = "hclust")
# Set a seed for reproducibility
set.seed(123)
# Create an 80-20 split
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
# Split the data
train_data <- predictors[train_index, ]
test_data <- predictors[-train_index, ]
train_target <- target[train_index]
test_target <- target[-train_index]
# Verify dimensions
cat("Training Set Dimensions (Predictors):", dim(train_data), "\n")
cat("Test Set Dimensions (Predictors):", dim(test_data), "\n")
cat("Training Set Dimensions (Target):", length(train_target), "\n")
cat("Test Set Dimensions (Target):", length(test_target), "\n")
# Remove highly correlated predictors from training and test sets
train_data_reduced <- train_data[, -tooHigh]
test_data_reduced <- test_data[, -tooHigh]
# Verify dimensions after removal
cat("Training Set Dimensions (After Removal):", dim(train_data_reduced), "\n")
cat("Test Set Dimensions (After Removal):", dim(test_data_reduced), "\n")
train_target <- as.factor(train_target)
test_target <- as.factor(test_target)
# Check levels of train_predictions and train_target
print("Levels of test_target:")
print(levels(test_target))
print("Levels of train_target:")
print(levels(train_target))
## Logistic Regression
library(caret)
# Set a seed for reproducibility
set.seed(123)
# Resampling using 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Training the model
logistic_model <- train(
  x = train_data,
  y = train_target,
  method = "glm",
  metric = "Kappa",
  family = binomial,
  trControl = ctrl
)
# Printing and plotting the results
print(logistic_model)
# Predicting on the training set
train_predictions <- predict(logistic_model, newdata = train_data)
# Creating Confusion Matrix for training predictions
train_conf_matrix <- confusionMatrix(
  data = train_predictions,
  reference = train_target,
  positive = "Yes" )
print(train_conf_matrix)
# Predicting on the test set
test_predictions <- predict(logistic_model, newdata = test_data)
# Creating Confusion Matrix for test predictions
test_conf_matrix <- confusionMatrix(
  data = test_predictions,
  reference = test_target,
  positive = "Yes" )
print(test_conf_matrix)
## Linear Discriminant Analysis
library(caret)
# Resampling using 10-fold cross-validation
lda_ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Set seed for reproducibility
set.seed(123)
# Train the LDA Model
lda_model <- train(
  x = train_data_reduced,
  y = train_target,
  method = "lda",
  metric = "Kappa",
  preProcess = c("center", "scale"),
  trControl = lda_ctrl
)
# Print model details
print(lda_model)
# Predict on the training set
lda_train_predictions <- predict(lda_model, newdata = train_data_reduced)
# Confusion Matrix for the Training Set
lda_train_conf_matrix <- confusionMatrix(
  data = lda_train_predictions,
  reference = train_target,
  positive = "Yes" )
print(lda_train_conf_matrix)
# Predict on the testing set
lda_test_predictions <- predict(lda_model, newdata = test_data_reduced)
# Confusion Matrix for the Testing Set
lda_test_conf_matrix <- confusionMatrix(
  data = lda_test_predictions,
  reference = test_target,
  positive = "Yes")
print(lda_test_conf_matrix)
## Partial Least Squar (PLS)
install.packages("pls")
library(pls)
# Resampling using 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Train the PLSDA model
set.seed(123)
plsda_model <- train(
  x = train_data,
  y = train_target,
  method = "pls",
  tuneGrid = expand.grid(.ncomp = 1:30),
  preProcess = c("center", "scale"),
  metric = "Kappa",
  trControl = ctrl
)
# Print model summary
print(plsda_model)
# Plot model performance
plot(plsda_model)
# Predict on the training set
plsda_train_predictions <- predict(plsda_model, newdata = train_data)
# Confusion Matrix for the Training Set
plsda_train_conf_matrix <- confusionMatrix(
  data = plsda_train_predictions,
  reference = train_target,
  positive = "Yes" )
print(plsda_train_conf_matrix)
# Predict on the testing set
plsda_test_predictions <- predict(plsda_model, newdata = test_data)
# Confusion Matrix for the Testing Set
plsda_test_conf_matrix <- confusionMatrix(
  data = plsda_test_predictions,
  reference = test_target,
  positive = "Yes")
print(plsda_test_conf_matrix)
## Penalized Model
install.packages("glmnet")
library(glmnet)
# Resampling using 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Define grid for hyperparameters
glmnet_grid <- expand.grid(
  .alpha = c(0, 0.1, 0.2, 0.4, 0.6, 0.8, 1),
  .lambda = seq(0.01, 0.2, length = 10)
)
# Set seed for reproducibility
set.seed(123)
# Train the glmnet model
glmnet_model <- train(
  x = train_data,
  y = train_target,
  method = "glmnet",
  tuneGrid = glmnet_grid,
  preProcess = c("center", "scale"),
  metric = "Kappa",
  trControl = ctrl
)
# Print the trained model
print(glmnet_model)
# Plot
plot(glmnet_model)
# Predict on the training set
train_predictions <- predict(glmnet_model, newdata = train_data)
# Confusion Matrix for the Training Set
train_conf_matrix <- confusionMatrix(
  data = train_predictions,
  reference = train_target,
  positive = "Yes" )
print(train_conf_matrix)
# Predict on the testing set
test_predictions <- predict(glmnet_model, newdata = test_data)
# Confusion Matrix for the Test Set
test_conf_matrix <- confusionMatrix(
  data = test_predictions,
  reference = test_target,
  positive = "Yes" )
print(test_conf_matrix)
## Quadratic Discriminant Analysis (QDA)
# Resampling using 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Set seed for reproducibility
set.seed(476)
# Train the QDA model
qdaFit <- train(
  x = train_data_reduced,
  y = train_target,
  method = "qda",
  metric = "Kappa",
  preProcess = c("center", "scale"),
  trControl = ctrl
)
# Print model results
print(qdaFit)
# Predict on the training set
qda_train_pred <- predict(qdaFit, newdata = train_data_reduced)
# Confusion Matrix for the Training Set
qda_train_conf <- confusionMatrix(
  data = qda_train_pred,
  reference = train_target,
  positive = "Yes")
print(qda_train_conf)
# Predict on the testing set
qda_test_pred <- predict(qdaFit, newdata = test_data_reduced)
# Confusion Matrix for the Test Set
qda_test_conf <- confusionMatrix(
  data = qda_test_pred,
  reference = test_target,
  positive = "Yes")
print(qda_test_conf)
## Regularized Discriminant Analysis (RDA)
install.packages("klaR")
library(klaR)
# Resampling using 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Set seed for reproducibility
set.seed(476)
# Train the RDA model
rdaFit <- train(
  x = train_data_reduced,
  y = train_target,
  method = "rda",
  metric = "Kappa",
  tuneGrid = expand.grid(
    .gamma = seq(0.01, 1, length = 20),
    .lambda = seq(0.01, 1, length = 20)
  ),
  preProcess = c("center", "scale"),
  trControl = ctrl
)
# Print model results
print(rdaFit)
# Plot model performance
plot(rdaFit)
# Predict on the training set
rda_train_pred <- predict(rdaFit, newdata = train_data_reduced)
#Confusion Matrix for the Training Set
rda_train_conf <- confusionMatrix(
  data = rda_train_pred,
  reference = train_target,
  positive = "Yes"
)
print(rda_train_conf)
#Predict on the testing set
rda_test_pred <- predict(rdaFit, newdata = test_data_reduced)
#Confusion Matrix for the Test Set
rda_test_conf <- confusionMatrix(
  data = rda_test_pred,
  reference = test_target,
  positive = "Yes")
print(rda_test_conf)
### Mixture Discriminant Analysis
install.packages("mda")
library(mda)
# Resampling using 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = defaultSummary
)
# Set seed for reproducibility
set.seed(476)
# Train the MDA model
mdaFit <- train(
  x = train_data_reduced,
  y = train_target,
  method = "mda",
  metric = "Kappa",
  tuneGrid = expand.grid(.subclasses = 1:30),
  preProcess = c("center", "scale"),
  trControl = ctrl
)
# Print model results
print(mdaFit)
# Plot
plot(mdaFit)
# Predict on the training set
mda_train_pred <- predict(mdaFit, newdata = train_data_reduced)
# Confusion Matrix for the Training Set
mda_train_conf <- confusionMatrix(
  data = mda_train_pred,
  reference = train_target,
  positive = "Yes" )
print(mda_train_conf)
# Predict on the testing set
mda_test_pred <- predict(mdaFit, newdata = test_data_reduced)
# Confusion Matrix for the Test Set
mda_test_conf <- confusionMatrix(
  data = mda_test_pred,
  reference = test_target,
  positive = "Yes")
print(mda_test_conf)
## Neural Networks
install.packages("nnet")
library(nnet)
#grid for tuning parameters
nnetGrid <- expand.grid(
  .size = 1:25,
  .decay = c(0, 0.01, 0.1, 0.5, 1, 2)
)
#maximum weights for neural network
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (ncol(train_data_reduced) + 1) + (maxSize + 1) * length(unique(train_target)))
#cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE,
  savePredictions = "final"
)
# Set seed for reproducibility
set.seed(123)
# Train the neural network model
nnetFit <- train(
  x = train_data_reduced,
  y = train_target,
  method = "nnet",
  metric = "Kappa",
  preProcess = c("center", "scale")
  tuneGrid = nnetGrid,
  trace = FALSE,
  maxit = 2000,
  trControl = ctrl
)
# Print and plot model results
print(nnetFit)
plot(nnetFit)
# Predict on the training set
train_predictions <- predict(nnetFit, newdata = train_data_reduced)
train_probabilities <- predict(nnetFit, newdata = train_data_reduced, type = "prob")
# Confusion Matrix for the Training Set
train_conf_matrix <- confusionMatrix(
  data = train_predictions,
  reference = train_target,
  positive = "Yes")
print(train_conf_matrix)
# Predict on the testing set
test_predictions <- predict(nnetFit, newdata = test_data_reduced)
test_probabilities <- predict(nnetFit, newdata = test_data_reduced, type = "prob")
# Confusion Matrix for the Testing Set
test_conf_matrix <- confusionMatrix(
  data = test_predictions,
  reference = test_target,
  positive = "Yes")
print(test_conf_matrix)
## Flexible Discriminant Analysis
#install the necessary packages
install.packages("MASS")
install.packages("mda")
install.packages("earth")
install.packages("themis")
#load the library
library(MASS)
library(mda)
library(earth)
library(themis)
#grid for hyperparameter tuning
marsGrid <- expand.grid(
  .degree = 1:2,
  .nprune = seq(2, 30)
)
# Set up cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Set seed for reproducibility
set.seed(123)
# Train the FDA model
fdaTuned <- train(
  x = train_data_reduced,
  y = train_target,
  method = "fda",
  tuneGrid = marsGrid,
  metric = "Kappa",
  preProcess = c("center", "scale"),
  trControl = ctrl
)
# Print and plot model results
print(fdaTuned)
plot(fdaTuned)
# Predict on the training set
fda_train_pred <- predict(fdaTuned, newdata = train_data_reduced)
fda_train_prob <- predict(fdaTuned, newdata = train_data_reduced, type = "prob")
# Confusion Matrix for the Training Set
fda_train_conf <- confusionMatrix(
  data = fda_train_pred,
  reference = train_target,
  positive = "Yes")
print(fda_train_conf)
# Predict on the testing set
fda_test_pred <- predict(fdaTuned, newdata = test_data_reduced)
fda_test_prob <- predict(fdaTuned, newdata = test_data_reduced, type = "prob")
# Confusion Matrix for the Testing Set
fda_test_conf <- confusionMatrix(
  data = fda_test_pred,
  reference = test_target,
  positive = "Yes" )
print(fda_test_conf)
## Support Vector Machines
# Install and load necessary package
install.packages("kernlab")
library(kernlab)
#sigma
sigmaValue <- sigest(as.matrix(train_data), frac = 0.9)[2]
#range of values for the Cost parameter
CValues <- 2^(seq(1, 10, by = 1))
# Create the tuning grid
svmRGrid <- expand.grid(.sigma = sigmaValue, .C = CValues)
# Set up 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE,
  savePredictions = "final"
)
# Set seed for reproducibility
set.seed(123)
# Train the SVM model
svmRModel <- train(
  x = train_data,
  y = train_target,
  method = "svmRadial",
  metric = "Kappa",
  preProcess = c("center", "scale"),
  tuneGrid = svmRGrid,
  trControl = ctrl
)
# Print and plot model results
print(svmRModel)
plot(svmRModel)
# Predict on the training set
train_pred <- predict(svmRModel, newdata = train_data)
train_prob <- predict(svmRModel, newdata = train_data, type = "prob")
# Confusion Matrix for the Training Set
train_conf <- confusionMatrix(
  data = train_pred,
  reference = train_target,
  positive = "Yes")
print(train_conf)
# Predict on the testing set
test_pred <- predict(svmRModel, newdata = test_data)
test_prob <- predict(svmRModel, newdata = test_data, type = "prob")
# Confusion Matrix for the Testing Set
test_conf <- confusionMatrix(
  data = test_pred,
  reference = test_target,
  positive = "Yes")
print(test_conf)
## K-Nearest Neighbors
library(caret)
# Set up 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE
)
# Set seed for reproducibility
set.seed(123)
# Train the KNN model
knnFit <- train(
  x = train_data,
  y = train_target,
  method = "knn",
  metric = "Kappa",
  preProcess = c("center", "scale"),
  tuneLength = 50,
  trControl = ctrl
)
# Print and plot model results
print(knnFit)
plot(knnFit, main = "KNN Tuning Results")
# Predict on the training set
train_pred <- predict(knnFit, newdata = train_data)
train_prob <- predict(knnFit, newdata = train_data, type = "prob")
# Confusion Matrix for the Training Set
train_conf <- confusionMatrix(
  data = train_pred,
  reference = train_target,
  positive = "Yes")
print(train_conf)
# Predict on the testing set
test_pred <- predict(knnFit, newdata = test_data)
test_prob <- predict(knnFit, newdata = test_data, type = "prob")
#Confusion Matrix for the Testing Set
test_conf <- confusionMatrix(
  data = test_pred,
  reference = test_target,
  positive = "Yes")
print(test_conf)
## Naive Bayes
#install necessary packages
install.packages("klaR")
#Load the library
library(klaR)
# Define the tuning grid for Naive Bayes
nbGrid_simple <- expand.grid(
  fL = c(1),
  usekernel = c(TRUE),
  adjust = c(1)
)
#Uisng 10 fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE,
  savePredictions = "final"
)
# Train the Naive Bayes model
set.seed(123)
nbFit <- train(
  x = train_data_reduced,
  y = train_target,
  method = "nb",
  metric = "Kappa",
  preProcess = c("BoxCox"),
  tuneGrid = nbGrid_simple,
  trControl = ctrl
)
# Print and visualize results
print(nbFit)
#plot(nbFit)
# Predict on the training set
train_pred <- predict(nbFit, newdata = train_data_reduced)
train_prob <- predict(nbFit, newdata = train_data_reduced, type = "prob")
# Confusion Matrix for the Training Set
train_conf <- confusionMatrix(
  data = train_pred,
  reference = train_target,
  positive = "Yes")
print(train_conf)
# Predict on the testing set
test_pred <- predict(nbFit, newdata = test_data_reduced)
test_prob <- predict(nbFit, newdata = test_data_reduced, type = "prob")
# Confusion Matrix for the Testing Set
test_conf <- confusionMatrix(
  data = test_pred,
  reference = test_target,
  positive = "Yes" )
print(test_conf)
## Important Predictors
#Install Necessary Libraraies
install.packages("caret")
library(caret)
#Install necessary Libraraies
install.packages("vip")
library(vip)
#Logistic Regression Predcitor Importance
variable_importance=varImp(logistic_model)
variable_importance
plot(variable_importance)
#MDA Predictor Importance
importance <- varImp(mdaFit)
importance
plot(importance)