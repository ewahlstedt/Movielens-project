##### Loading required libraries #####
library(caret)
library(data.table)
library(dplyr)
library(FNN)
library(glmnet)
library(ggrepel)
library(h2o)
library(knitr)
library(lubridate)
library(speedglm)
library(tidyverse)
library(xgboost)

##### Run code provided for course to set up edx and final_holdout_test sets #####

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
#set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##### Start of my own code #####

##### Data exploration and tidying of data #####

# Inspect edx dataset structure
head(edx)

# Change format of timestamp to numeric and round to closest month
edx$timestamp <- as.numeric(round_date(as_datetime(edx$timestamp), "month"))

# Tidy up titles column to extract the release year of each movie
edx_tidy <- edx %>% separate(title, c("title", "release_year"), -6) # Get the release year, but it's still in brackets

release_year_new <- str_extract(edx_tidy$release_year, "\\d{4}") # Remove brackets

edx_tidy$release_year <- as.numeric(release_year_new) # Overwrite the column in main dataset with the one without brackets

rm(release_year_new) # Remove as no longer needed

# Check the structure of edx_tidy
head(edx_tidy)

#Make an overview table of edx dataset
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

summary_table_edx <- edx %>%
  summarise(
    `Number of ratings` = n(),
    `Number of unique movies` = n_distinct(movieId),
    `Number of unique users` = n_distinct(userId),
    `Number of unique genre combinations` = n_distinct(genres),
    `Mean of ratings` = mean(rating, na.rm = TRUE),
    `Median of ratings` = median(rating, na.rm = TRUE),
    `Mode of ratings` = get_mode(rating)
  )

# Convert the summary table to long format
summary_table_edx_long <- summary_table_edx %>%
  pivot_longer(cols = everything(), names_to = "Statistic", values_to = "Value") %>%
  mutate(Value = case_when(
    Statistic %in% c("Mean rating", "Median rating", "Mode rating") ~ format(round(Value, 1), nsmall = 1), # Set significant digits for mean and median
    TRUE ~ format(round(Value, 0), nsmall = 0) # Set significant digits for other rows
  ))

kable(summary_table_edx_long, caption = "Overview of the edx dataset")


# Check for missing values in edx and final_holdout_test datasets
mean(is.na(edx) == "TRUE") # This comes to 0, meaning there are no missing values in our dataset.
mean(is.na(final_holdout_test == "TRUE")) # This also comes to 0.

##### MovieId and UserId #####

edx %>% group_by(userId) %>% summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(fill = "sienna2") +
  labs(x = "Mean rating", y = "Number of users", title = "Average user ratings")

edx %>% group_by(movieId) %>% summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(fill = "palegreen3") +
  labs(x = "Mean rating", y = "Number of movies", title = "Average movie ratings")

edx %>% group_by(userId) %>% summarize(sd_rating = sd(rating)) %>%
  ggplot(aes(sd_rating)) +
  geom_histogram(fill = "sienna2") +
  labs(x = "Standard deviation of rating", y = "Number of users", title = "Standard deviation of user ratings")

edx %>% group_by(movieId) %>% summarize(sd_rating = sd(rating)) %>%
  ggplot(aes(sd_rating)) +
  geom_histogram(fill = "palegreen3") +
  labs(x = "Standard deviation of rating", y = "Number of movies", title = "Standard deviation of movie ratings")

# Set up new predictors
edx_tidy <- edx_tidy %>%
  group_by(userId) %>% # Create variables by userId
  mutate(user_avg_rating = mean(rating),
         user_rating_count = n(),
         user_sd_rating = sd(rating)) %>%
  ungroup() %>%
  group_by(movieId) %>% # Create varables by movieId
  mutate(movie_avg_rating = mean(rating),
         movie_rating_count = n(),
         movie_sd_rating = sd(rating)) %>%
  ungroup()

edx_tidy$movie_sd_rating[is.na(edx_tidy$movie_sd_rating)] <- 0 # Fill any missing values with 0 (movies with only one rating have no SD)
edx_tidy$user_sd_rating[is.na(edx_tidy$user_sd_rating)] <- 0 # Fill any missing values with 0 (users that have rated only one movie have no SD)

setDT(edx_tidy) # Back to data.table as above operations may convert to data.frame
head(edx_tidy) # Check that all columns are there

##### Movie Genres ##### 

# Turn genres variable into separate "dummy" variables for each genres
# Separate rows by genres
edx_expanded <- edx_tidy[, .(genre = unlist(strsplit(as.character(genres), "\\|"))),
                    by = .(userId, movieId, release_year, title, rating, timestamp, user_avg_rating, user_rating_count, user_sd_rating, movie_avg_rating, movie_rating_count, movie_sd_rating)]

# Create genre_indicator column
edx_expanded[, genre_indicator := 1]

# Pivot to wide format
edx_wide <- dcast(edx_expanded, userId + movieId + title + release_year + rating + timestamp + user_avg_rating + user_rating_count + user_sd_rating + movie_avg_rating + movie_rating_count + movie_sd_rating ~ genre,
                  value.var = "genre_indicator", fill = 0)

#rm(edx_expanded) # Remove as no longer needed

setDT(edx_wide) # Set to data.table again as some of the operations above may have converted the data back into a data.frame

# Select unique movies by movieId
unique_movies <- unique(edx_wide, by = "movieId")

# Select only the genre columns
genre_columns <- setdiff(names(unique_movies), c("userId", "movieId", "title", "release_year", "rating", "timestamp", "user_avg_rating", "user_rating_count", "user_sd_rating", "movie_avg_rating", "movie_rating_count", "movie_sd_rating"))

# Sum occurrences of each genre across unique movies
Count <- colSums(unique_movies[, .SD, .SDcols = genre_columns])
Count <- data.frame(Count)
Genre <- rownames(Count)
genres <- cbind(Genre, data.frame(Count, row.names=NULL))
genres <- genres %>% arrange(desc(Count))

# Table of genres and number of times they have been assigned to a movieId (only using unique movieIds to avoid counting gernes multiple times for same movie)
kable(genres, caption = "Number of movies within each genre (most movies appear in more than one genre)")

setDT(edx) # Set to data.table again as some of the operations above may have converted the data back into a data.frame

# Inspect the one movie with no genres
edx[genres == "(no genres listed)"]

# Add prefix 'genre_' to all genre columns to allow for easier str_detect and such later if relevant
genre_columns <- setdiff(names(edx_wide), c("userId", "movieId", "title", "release_year", "rating", "timestamp", "user_avg_rating", "user_rating_count", "user_sd_rating", "movie_avg_rating", "movie_rating_count", "movie_sd_rating"))
new_genre_columns <- paste("genre", genre_columns, sep = "_")
edx_wide <- setnames(edx_wide, old = genre_columns, new = new_genre_columns)

# Change problematic column names that do not fit well with R syntax
colnames(edx_wide)[which(names(edx_wide) == "genre_(no genres listed)")] <- "genre_no_genres_listed"
colnames(edx_wide)[which(names(edx_wide) == "genre_Film-Noir")] <- "genre_Film_Noir"


##### Making the dataset more lean - selecting predictors #####

# Title column removed as will not be useful for analysis (adds no more info than the more concise movieId)
edx_wide <- edx_wide[, !"title", with = FALSE]

# Check for collinearity using Correlation Matrix - to see if any predictors are closely related and if so only one can be kept
cor_matrix_wide <- cor(edx_wide[, !"rating", with = FALSE])
findCorrelation(cor_matrix_wide)


#### Creating train and test datasets from the edx_wide dataset in preparation for random forest importance check and for later testing of models

# Create train and test sets from edx_wide - using same methodology as used for creation of edx and final_holdout_test sets above
set.seed(1) # Set seed to ensure reproducability

edx_test_index_wide <- createDataPartition(y = edx_wide$rating, times = 1, p = 0.1, list = FALSE) # Use same partition as above to stay consistent

edx_train_wide <- edx_wide[-edx_test_index_wide,]

temp <- edx_wide[edx_test_index_wide,]

# Make sure userId and movieId in final hold-out test set are also in edx_train set

edx_test_wide <- temp %>%
  semi_join(edx_train_wide, by = "movieId") %>%
  semi_join(edx_train_wide, by = "userId")

# Add rows removed from edx_test set back into edx_train set

removed <- anti_join(temp, edx_test_wide)

edx_train_wide <- rbind(edx_train_wide, removed)

rm(edx_test_index_wide, temp, removed) # Remove as not needed

setDT(edx_train_wide) # Covert to data.table format if not already
setDT(edx_test_wide) # Covert to data.table format if not already


##### Run random forest model with all 30 predictors to gain insight into predictor importance #####

# Initialize the H2O cluster
h2o.init(nthreads = -1, max_mem_size = "4G")

# Convert data to h2o frames
train_h2o <- as.h2o(edx_train_wide)
test_h2o <- as.h2o(edx_test_wide)

# Define the response and predictor variables
target <- "rating"
predictors <- setdiff(names(edx_train_wide), target)

# Train the random forest model using h2o
model <- h2o.randomForest(
  y = target,
  x = predictors,
  training_frame = train_h2o,
  ntrees = 100,
  mtries = 3,
  min_rows = 5,
  seed = 1
)

# Predict on the test set
predictions <- h2o.predict(model, test_h2o)

# Extract variable importance
importance <- h2o.varimp(model)

# Shutdown H2O cluster
h2o.shutdown(prompt = FALSE)

# Table with ten most important predictors based on random forest model
kable(head(importance, 10), caption = "The top ten most important predictors by importance")

##### Keep only five most important predictors along with rating, userId and movieId #####

# Remove columns from the full edx_wide dataset
edx_final <- edx_wide[, .(rating, userId, movieId, movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)]

# Remove columns from train and test sets
edx_train_final <- edx_train_wide[, .(rating, userId, movieId, movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)]
edx_test_final <- edx_test_wide[, .(rating, userId, movieId, movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)]

setDT(edx_final) # Covert to data.table format if not already
setDT(edx_train_final) # Covert to data.table format if not already
setDT(edx_test_final) # Covert to data.table format if not already

#Check edx_final
head(edx_final)


##### Finding the optimal model #####

# Write function to calculate RMSE
RMSE <- function(predicted_ratings, true_ratings){sqrt(mean((predicted_ratings - true_ratings)^2))}


##### The course method #####

# Set up the model like in the course

# Calculate the mean rating of all movies
mean_ratings <- mean(as.numeric(edx_train_final$rating))

# Calculate the RMSE based on mean only, for comparison
mean_only_rmse <- RMSE(mean_ratings, edx_test_final$rating)

# Calculate the effects of each predictor
movie_avg <- edx_train_final %>%
  group_by(movie_avg_rating) %>%
  summarize(b_m_a = mean(rating - mean_ratings))

movie_sd <- edx_train_final %>%
  group_by(movie_sd_rating) %>%
  summarize(b_m_sd = mean(rating - mean_ratings))

movie_count <- edx_train_final %>%
  group_by(movie_rating_count) %>%
  summarize(b_m_c = mean(rating - mean_ratings))

user_avg <- edx_train_final %>%
    group_by(user_avg_rating) %>%
  summarize(b_u_a = mean(rating - mean_ratings))

user_sd <- edx_train_final %>%
    group_by(user_sd_rating) %>%
  summarize(b_u_sd = mean(rating - mean_ratings))

# Predicted ratings with all five predictors
pred_course_all <- edx_test_final %>%
  left_join(movie_avg, by='movie_avg_rating') %>%
  left_join(movie_sd, by='movie_sd_rating') %>%
  left_join(movie_count, by='movie_rating_count') %>%
  left_join(user_avg, by='user_avg_rating') %>%
  left_join(user_sd, by='user_sd_rating') %>%
  mutate(pred = mean_ratings + b_m_a + b_m_sd + b_m_c + b_u_a + b_u_sd) %>%
  pull(pred)

# Predicted ratings with just the top two predictors (movie_avg_rating and user_avg_rating)
pred_course_top_two <- edx_test_final %>%
  left_join(movie_avg, by='movie_avg_rating') %>%
  left_join(user_avg, by='user_avg_rating') %>%
  mutate(pred = mean_ratings + b_m_a + b_u_a) %>%
  pull(pred)

# Calculate the RMSEs
course_all_rmse <- RMSE(pred_course_all, edx_test_final$rating)
course_top_two_rmse <- RMSE(pred_course_top_two, edx_test_final$rating)

# Create table of the RMSEs
course_rmses <- data.frame(
  Model = c("Mean only", "Course model - all predictors", "Course model - top two predictors"),
  RMSE = c(mean_only_rmse, course_all_rmse, course_top_two_rmse)
)
kable(course_rmses, caption = "RMSEs obtained with course model")


##### Regularisation #####

# Apply lambda to the full dataset
lambda <- 0.1 # This lambda was found manually as cross validation crashed R - values tried manually were 0.1, 1 and 10. The best result was aciheved using 0.1.

# Calculate overall mean rating
overall_mean_rating <- mean(edx_train_final$rating)

# Calculate effects of individual predictors with regularisation
movie_avg <- edx_train_final %>%
  group_by(movie_avg_rating) %>%
  summarize(b_m_a = sum(rating - overall_mean_rating) / (n() + lambda))

movie_sd <- edx_train_final %>%
  group_by(movie_sd_rating) %>%
  summarize(b_m_sd = sum(rating - overall_mean_rating) / (n() + lambda))

movie_count <- edx_train_final %>%
  group_by(movie_rating_count) %>%
  summarize(b_m_c = sum(rating - overall_mean_rating) / (n() + lambda))

user_avg <- edx_train_final %>%
  group_by(user_avg_rating) %>%
  summarize(b_u_a = sum(rating - overall_mean_rating) / (n() + lambda))

user_sd <- edx_train_final %>%
  group_by(user_sd_rating) %>%
  summarize(b_u_sd = sum(rating - overall_mean_rating) / (n() + lambda))

# Predicted ratings with all five predictors
pred_reg_all <- edx_test_final %>%
  left_join(movie_avg, by='movie_avg_rating') %>%
  left_join(movie_sd, by='movie_sd_rating') %>%
  left_join(movie_count, by='movie_rating_count') %>%
  left_join(user_avg, by='user_avg_rating') %>%
  left_join(user_sd, by='user_sd_rating') %>%
  mutate(pred = overall_mean_rating + b_m_a + b_m_sd + b_m_c + b_u_a + b_u_sd) %>%
  pull(pred)

# Predicted ratings with just the top two predictors (movie_avg_rating and user_avg_rating)
pred_reg_top_two <- edx_test_final %>%
  left_join(movie_avg, by='movie_avg_rating') %>%
  left_join(user_avg, by='user_avg_rating') %>%
  mutate(pred = overall_mean_rating + b_m_a + b_u_a) %>%
  pull(pred)

# Calculate RMSEs
reg_all_rmse <- RMSE(pred_reg_all, edx_test_final$rating)

reg_top_two_rmse <- RMSE(pred_reg_top_two, edx_test_final$rating)

# Create table with RMSEs
reg_rmses <- data.frame(
  Model = c("Regularisation - all predictors", "Regularisation - top two predictors"),
  RMSE = c(reg_all_rmse, reg_top_two_rmse)
)
kable(reg_rmses, caption = "RMSEs obtained with regularisation model")


##### k Nearest Neighbour (kNN) model #####

# Scale the data for k-nn modelling
# Create a method for preprocessing the data
preprocess_method <- function(train_data, test_data) {
  preprocess_options <- preProcess(train_data, method = c("center", "scale"))
  train_scaled <- predict(preprocess_options, train_data)
  test_scaled <- predict(preprocess_options, test_data)
  return(list(train = train_scaled, test = test_scaled, preProc = preprocess_options))
}

# Exclude the target variable 'rating' and userId and movieId, as thesewill not be used for prediction

train_predictors <- edx_train_final[, .(movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)]
test_predictors <- edx_test_final[, .(movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)]

# Apply the preprocessing function

preprocessed_data <- preprocess_method(train_predictors, test_predictors)

edx_train_final_scaled <- preprocessed_data$train
edx_test_final_scaled <- preprocessed_data$test

# Add the target variable back to the scaled data

edx_train_final_scaled$rating <- edx_train_final$rating
edx_test_final_scaled$rating <- edx_test_final$rating

setDT(edx_train_final_scaled) # Covert to data.table format if not already
setDT(edx_test_final_scaled) # Covert to data.table format if not already

# Use scale data for the k-nn model
# Perform grid search for the best k
k_values <- seq(from = 80, to = 110, by = 5)
rmse_values <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  knn_model <- knn.reg(train = edx_train_final_scaled[, -"rating", with = FALSE], 
                       test = edx_test_final_scaled[, -"rating", with = FALSE], 
                       y = edx_train_final_scaled$rating, k = k_values[i])
  
  pred_knn <- knn_model$pred
  
  # Calculate RMSE
  rmse_values[i] <- RMSE(pred_knn, edx_test_final_scaled$rating)
}

# Find the best k - only did this once due to processing constraints, ideally would be done for both all predictor model and top two predictor model.
best_k <- k_values[which.min(rmse_values)]

# Perform kNN using FNN - all predictors
knn_model_all <- knn.reg(train = edx_train_final_scaled[, -"rating", with = FALSE], 
                     test = edx_test_final_scaled[, -"rating", with = FALSE], 
                     y = edx_train_final_scaled$rating, k = best_k)

# Perform kNN using FNN - top two predictors
knn_model_top_two <- knn.reg(train = edx_train_final_scaled[, -c("rating", "movie_sd_rating", "user_sd_rating", "movie_rating_count")], 
                              test = edx_test_final_scaled[, -c("rating", "movie_sd_rating", "user_sd_rating", "movie_rating_count")], 
                              y = edx_train_final_scaled$rating, 
                              k = best_k)

# Extract predictions
pred_knn_all <- knn_model_all$pred
pred_knn_top_two <- knn_model_top_two$pred

# Calculate RMSE
knn_rmse_all <- RMSE(pred_knn_all, edx_test_final_scaled$rating)
knn_rmse_top_two <- RMSE(pred_knn_top_two, edx_test_final_scaled$rating)

# Create table of RMSEs
knn_rmses <- data.frame(
  Model = c("k-nn model - all predictors", "k-nn model - top two predictors"),
  RMSE = c(knn_rmse_all, knn_rmse_top_two)
)
kable(knn_rmses, caption = "RMSEs obtained with k-nn model")


##### Random Forest model #####
# Run a random forest model using h2o, similar to before but this time only using the dataset with selected predictors

# Initialize the H2O cluster
h2o.init(nthreads = -1, max_mem_size = "4G") # Set memory size to accommodate the model

### All five predictors ###

# Convert data to H2O frames - all predictors
train_h2o_all <- as.h2o(edx_train_final[, .(rating, movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)])
test_h2o_all <- as.h2o(edx_test_final[, .(rating, movie_avg_rating, user_avg_rating, movie_sd_rating, movie_rating_count, user_sd_rating)])

# Define the response and predictor variables - all predictors
target_all <- "rating"
predictors_all <- c("movie_avg_rating", "user_avg_rating", "movie_sd_rating", "movie_rating_count", "user_sd_rating")

# Train the random forest model using H2O - all predictors
model_h2o_all <- h2o.randomForest(
  y = target_all,
  x = predictors_all,
  training_frame = train_h2o_all,
  ntrees = 100,     
  mtries = -1,      # Automatically choose mtries
  min_rows = 20,    # Increase min_rows to reduce memory usage
  max_depth = 20,   # Decrease max_depth to limit tree complexity
  seed = 1
)

# Predict on the test set - all predictors
pred_rf_all <- h2o.predict(model_h2o_all, test_h2o_all)

# Extract the actual and predicted values - all predictors
actual_values_all <- as.numeric(edx_test_final$rating)

# Convert H2O frame to R data frame and extract predictions - all predictors
pred_rf_all_df <- as.data.frame(pred_rf_all)
predicted_values_all <- as.vector(pred_rf_all_df$predict)

# Calculate RMSE - all predictors
rf_rmse_all <- RMSE(predicted_values_all, actual_values_all)

### Top two predictors ###

# Convert data to H2O frames - top two predictors
train_h2o_top_two <- as.h2o(edx_train_final[, .(rating, movie_avg_rating, user_avg_rating)])
test_h2o_top_two <- as.h2o(edx_test_final[, .(rating, movie_avg_rating, user_avg_rating)])

# Define the response and predictor variables - top two predictors
target_top_two <- "rating"
predictors_top_two <- c("movie_avg_rating", "user_avg_rating")

# Train the random forest model using H2O - top two predictors
model_h2o_top_two <- h2o.randomForest(
  y = target_top_two,
  x = predictors_top_two,
  training_frame = train_h2o_top_two,
  ntrees = 100,     
  mtries = -1,      # Automatically choose mtries
  min_rows = 20,    # Increase min_rows to reduce memory usage
  max_depth = 20,   # Decrease max_depth to limit tree complexity
  seed = 1
)

# Predict on the test set - top two predictors
pred_rf_top_two <- h2o.predict(model_h2o_top_two, test_h2o_top_two)

# Extract the actual and predicted values  - top two predictors
actual_values_top_two <- as.vector(edx_test_final$rating)

# Convert H2O frame to R data frame and extract predictions  - top two predictors
pred_rf_top_two_df <- as.data.frame(pred_rf_top_two)
predicted_values_top_two <- as.vector(pred_rf_top_two_df$predict)

# Calculate RMSE - top two predictors
rf_rmse_top_two <- RMSE(predicted_values_top_two, actual_values_top_two)

# Shutdown H2O cluster
h2o.shutdown(prompt = FALSE)

# Calculate RMSE - top two predictors
rf_rmse_top_two <- RMSE(pred_rf_all, edx_test_final$rating)

# Create table with RMSEs
rf_rmses <- data.frame(
  Model = c("pred_rf - all predictors", "pred_rf - top two predictors")
  RMSE = c(rf_rmse_all, rf_rmse_top_two)
)
kable(rf_rmses, caption = "RMSEs obtained with random forest model")


##### speedglm model #####
# This model benegits from scaled data, so the scaled datasets created for k-nn model above is used.
# Fit the model using scaled data - using both all predictors and top two predictors
model_speedglm_all <- speedglm(rating ~ movie_avg_rating + user_avg_rating + movie_sd_rating + movie_rating_count + user_sd_rating, data = edx_train_final_scaled)

model_speedglm_top_two <- speedglm(rating ~ movie_avg_rating + user_avg_rating, data = edx_train_final_scaled)

# Predict on test set
pred_speedglm_all <- predict(model_speedglm_all, edx_test_final_scaled)

pred_speedglm_top_two <- predict(model_speedglm_top_two, edx_test_final_scaled)

# Calculate RMSEs
speedglm_rmse_all <- RMSE(pred_speedglm_all, edx_test_final_scaled$rating)
speedglm_rmse_top_two <- RMSE(pred_speedglm_top_two, edx_test_final_scaled$rating)

# Create table with RMSEs
speedglm_rmses <- data.frame(Model = c("speedglm - all predictors", "speedglm - top two predictors"), RMSE = c(speedglm_rmse_all, speedglm_rmse_top_two))
kable(speedglm_rmses, caption = "RMSEs obtained with speedglm model")


#### xgboost model

# Convert training data to xgb.DMatrix - all predictors
dtrain_all <- xgb.DMatrix(
  data = as.matrix(edx_train_final[, c("movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "movie_rating_count"), with = FALSE]),
  label = edx_train_final$rating
)

# Convert test data to xgb.DMatrix - all predictors
dtest_all <- xgb.DMatrix(
  data = as.matrix(edx_test_final[, c("movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "movie_rating_count"), with = FALSE])
)

# Convert training data to xgb.DMatrix - top two predictors
dtrain_top_two <- xgb.DMatrix(
  data = as.matrix(edx_train_final[, c("movie_avg_rating", "user_avg_rating"), with = FALSE]),
  label = edx_train_final$rating
)

# Convert test data to xgb.DMatrix - top two predictors
dtest_top_two <- xgb.DMatrix(
  data = as.matrix(edx_test_final[, c("movie_avg_rating", "user_avg_rating"), with = FALSE])
)

# Fit the model with the correct parameters - all predictors
params <- list(objective = "reg:squarederror")
model_xgboost_all <- xgboost(data = dtrain_all, params = params, nrounds = 300)

# Fit the model with the correct parameters - top_two predictors
params <- list(objective = "reg:squarederror")
model_xgboost_top_two <- xgboost(data = dtrain_top_two, params = params, nrounds = 300)

# Predict on test data - all predictors
pred_xgboost_all <- predict(model_xgboost_all, dtest_all)

# Predict on test data - top two predictors
pred_xgboost_top_two <- predict(model_xgboost_top_two, dtest_top_two)

# Calculate RMSE - all predictors
xgboost_rmse_all <- sqrt(mean((pred_xgboost_all - edx_test_final$rating)^2))

# Calculate RMSE - top two predictors
xgboost_rmse_top_two <- sqrt(mean((pred_xgboost_top_two - edx_test_final$rating)^2))

# Create table with RMSEs 
xgboost_rmses <- data.frame(
  Model = c("xgboost - all predictors", "xgboost - top two predictors"),
  RMSE = c(xgboost_rmse_all, xgboost_rmse_top_two)
)
kable(xgboost_rmses, caption = "RMSEs obtained with xgboost model")


##### Results #####

# Table of all model RMSEs so far
rmse_table <- data.frame(
  Model = c("Course model - all predictors", "Course model - top two predictors", "Regularisation - all predictors", "Regularisation - top two predictors", "k-nn model - all predictors", "k-nn model - top two predictors", "random forest - all predictors", "random forest - top two predictors", "speedglm - all predictors", "speedglm - top two predictors", "xgboost - all predictors", "xgboost - top two predictors"),
  RMSE = c(course_all_rmse, course_top_two_rmse, reg_all_rmse, reg_top_two_rmse, knn_rmse_all, knn_rmse_top_two, rf_rmse_all, rf_rmse_top_two, speedglm_rmse_all, speedglm_rmse_top_two, xgboost_rmse_all, xgboost_rmse_top_two)
)
# Create the table
kable(arrange(rmse_table, desc(RMSE)), caption = "RMSE for Different Models")


##### Ensemble model #####

# Averaging method - only using this rather than training a model as R crashed repeatedly when trying

ensemble_avg <- (pred_rf_all_df$predict + pred_rf_top_two_df$predict + pred_xgboost_all) / 3

ensemble_avg_rmse <- RMSE(ensemble_avg, edx_test_final$rating)

# Update table above with ensemble RMSE
# Update table above with ensemble RMSE
rmse_table <- data.frame(
  Model = c("Course model - all predictors", "Course model - top two predictors", "Regularisation - all predictors", "Regularisation - top two predictors", "k-nn model - all predictors", "k-nn model - top two predictors", "random forest - all predictors", "random forest - top two predictors", "speedglm - all predictors", "speedglm - top two predictors", "xgboost - all predictors", "xgboost - top two predictors", "Ensemble model (averaging)"),
  RMSE = c(course_all_rmse, course_top_two_rmse, reg_all_rmse, reg_top_two_rmse, knn_rmse_all, knn_rmse_top_two, rf_rmse_all, rf_rmse_top_two, speedglm_rmse_all, speedglm_rmse_top_two, xgboost_rmse_all, xgboost_rmse_top_two, ensemble_avg_rmse)
)

kable(arrange(rmse_table, desc(RMSE)))


##### Final Results #####

# Add predictors to final_holdout_test as done to edx_final set
final_holdout_test <- final_holdout_test %>%
  group_by(userId) %>% # Create variables by userId
  mutate(user_avg_rating = mean(rating),
         user_rating_count = n(),
         user_sd_rating = sd(rating)) %>%
  ungroup() %>%
  group_by(movieId) %>% # Create varables by movieId
  mutate(movie_avg_rating = mean(rating),
         movie_rating_count = n(),
         movie_sd_rating = sd(rating)) %>%
  ungroup()

final_holdout_test$movie_sd_rating[is.na(final_holdout_test$movie_sd_rating)] <- 0 # Fill any missing values with 0 (movies with only one rating have no SD)
final_holdout_test$user_sd_rating[is.na(final_holdout_test$user_sd_rating)] <- 0 # Fill any missing values with 0 (users that have rated only one movie have no SD)

setDT(final_holdout_test) # Back to data.table as above operations may convert to data.frame
head(final_holdout_test) # Check that columns are included

# Run final model with edx_final as train set and final_holdout_test as test set
# Convert training data to xgb.DMatrix
dtrain <- xgb.DMatrix(
  data = as.matrix(edx_final[, c("movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "movie_rating_count"), with = FALSE]),
  label = edx_final$rating
)

# Convert test data to xgb.DMatrix
dtest <- xgb.DMatrix(
  data = as.matrix(final_holdout_test[, c("movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "movie_rating_count"), with = FALSE])
)

# Fit the model with the correct parameters
params <- list(objective = "reg:squarederror")
final_model_xgboost <- xgboost(data = dtrain, params = params, nrounds = 300)

# Predict on test data
final_pred_xgboost <- predict(final_model_xgboost, dtest)

# Calculate RMSE
final_rmse_xgboost <- RMSE(final_pred_xgboost, final_holdout_test$rating)


# The final RMSE is:  
print(final_rmse_xgboost) # Final RMSE
