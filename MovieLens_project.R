##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(stringr)

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

###############################
# Start of my own code
###############################

######### Packages loaded
library(lubridate)
library(dplyr)
library(ggrepel)
library(kknn)
library(randomForest)
library(data.table)
library(knitr)


######### Tidy and pre-process data ###############

# Ensure the data is a data.table for faster processing
setDT(edx)

# Are there any missing values in our dataset?
mean(is.na(edx) == "TRUE") # This comes to 0, meaning there are no missing values in our dataset.
mean(is.na(final_holdout_test == "TRUE")) # This also comes to 0.

#What are the predictors in the original edx dataset? We are looking to predict rating - what other columns do we have at our disposal?

head(edx)

# Tidy data to separate release year from the title
edx_tidy <- edx %>% separate(title, c("title", "release_year"), -6) # Get the release year, but it's still in brackets

release_year_new <- str_extract(edx_tidy$release_year, "\\d{4}") # Remove brackets

edx_tidy$release_year <- as.numeric(release_year_new) # Overwrite the column in main dataset with the one without brackets

rm(release_year_new) # Remove as no longer needed

# Title column removed as will not be useful for analysis
edx_tidy <- edx_tidy %>% select(-"title")

# Add columns for userId and movieId mean rating, number of ratings and sd of ratings, respectively
edx_tidy <- edx_tidy %>%
  group_by(userId) %>%
  mutate(user_avg_rating = mean(rating),
         user_rating_count = n(),
         user_sd_rating = sd(rating)) %>%
  ungroup() %>%
  group_by(movieId) %>%
  mutate(movie_avg_rating = mean(rating),
         movie_rating_count = n(),
         movie_sd_rating = sd(rating)) %>%
  ungroup()

edx_tidy$movie_sd_rating[is.na(edx_tidy$movie_sd_rating)] <- 0

setDT(edx_tidy) # Back to data.table as above operations may convert to data.frame

##Turn genres variable into separate columns for each genre
# Separate rows by genres
edx_expanded <- edx_tidy[, .(genre = unlist(strsplit(as.character(genres), "\\|"))),
                    by = .(userId, movieId, release_year, rating, timestamp, user_avg_rating, user_rating_count, user_sd_rating, movie_avg_rating, movie_rating_count, movie_sd_rating)]

# Create genre_indicator column
edx_expanded[, genre_indicator := 1]

# Pivot to wide format
edx_wide <- dcast(edx_expanded, userId + movieId + release_year + rating + timestamp + user_avg_rating + user_rating_count + user_sd_rating + movie_avg_rating + movie_rating_count + movie_sd_rating ~ genre,
                  value.var = "genre_indicator", fill = 0)

# Add prefix 'genre_' to all genre columns
genre_columns <- setdiff(names(edx_wide), c("userId", "movieId", "release_year", "rating", "timestamp", "user_avg_rating", "user_rating_count", "user_sd_rating", "movie_avg_rating", "movie_rating_count", "movie_sd_rating"))
new_genre_columns <- paste("genre", genre_columns, sep = "_")
edx_wide <- setnames(edx_wide, old = genre_columns, new = new_genre_columns)

# Change problematic column names
colnames(edx_wide)[which(names(edx_wide) == "genre_(no genres listed)")] <- "genre_no_genres_listed"
colnames(edx_wide)[which(names(edx_wide) == "genre_Film-Noir")] <- "genre_Film_Noir"
colnames(edx_wide)[which(names(edx_wide) == "genre_Sci-Fi")] <- "genre_Sci_Fi"

#Remove intermediate files
#rm(edx_expanded)

setDT(edx_wide)

##### Check if new features can add value ##########

# # Check for Collinearity using Correlation Matrix
# cor_matrix_wide <- cor(edx_wide %>% select(-rating))
# print(cor_matrix_wide)
# findCorrelation(cor_matrix_wide)
# 
# cor_matrix_tidy <- cor(edx_tidy %>% select(-rating, -genres))
# print(cor_matrix_tidy)
# findCorrelation(cor_matrix_tidy)
# 
# # Check for Collinearity using VIF
# library(car)
# vif_model_tidy <- lm(rating ~ . - userId - movieId - release_year - timestamp, data = edx_tidy)
# vif_values_tidy <- vif(vif_model)
# print(vif_values_tidy)
# 
# vif_model_wide <- lm(rating ~ . - userId - movieId - timestamp - release_year - user_avg_rating - user_rating_count - user_sd_rating - movie_avg_rating - movie_rating_count - movie_sd_rating, data = edx_wide)
# vif_values_wide <- vif(vif_model_wide)
# print(vif_values_wide)

############# Familiarise ourselves with data ##############

#######Do some summary statistics and overview of data ####

# Make an overview table of edx dataset
# get_mode <- function(v) {
#   uniqv <- unique(v)
#   uniqv[which.max(tabulate(match(v, uniqv)))]
# }
# 
# summary_table_edx <- edx %>%
#   summarise(
#     `Number of ratings` = n(),
#     `Number of unique movies` = n_distinct(movieId),
#     `Number of unique users` = n_distinct(userId),
#     `Number of unique genre combinations` = n_distinct(genres),
#     `Mean rating` = mean(rating, na.rm = TRUE),
#     `Median rating` = median(rating, na.rm = TRUE),
#     `Mode rating` = get_mode(rating)
#   )
# 
# # Convert the summary table to long format
# summary_table_edx_long <- summary_table_edx %>%
#   pivot_longer(cols = everything(), names_to = "Statistic", values_to = "Value") %>%
#   mutate(Value = case_when(
#     Statistic %in% c("Mean rating", "Median rating", "Mode rating") ~ format(round(Value, 1), nsmall = 1), # Set significant digits for mean and median
#     TRUE ~ format(round(Value, 0), nsmall = 0) # Set significant digits for other rows
#   ))
# 
# kable(summary_table_edx_long, caption = "Overview of the edx dataset")
# 
# # Genre names list
# genre_list <- unique(unlist(strsplit(as.character(edx$genres), "\\|")))
# kable(genre_list)


# Table with data quality check
# nzv_edx <- nearZeroVar(edx_wide)
# 
# summary_table_tidy <- edx %>%
#   summarise(
#     `Number of missing values` = is.na(edx),
#     `Variables with near-zero variance` = nzv_edx,
#   )
# 
# kable(summary_table_tidy)


#### Make a table with genres
# # Select unique movies by movieId
# unique_movies <- unique(edx_wide, by = "movieId")
# 
# # Select only the genre columns
# genre_columns <- setdiff(names(unique_movies), c("userId", "movieId", "title", "release_year", "years_since_release", "rating", "timestamp"))
# 
# # Sum occurrences of each genre across unique movies
# genre_counts <- colSums(unique_movies[, .SD, .SDcols = genre_columns])
# 
# # Calculate the total number of unique movies
# total_movies <- nrow(unique_movies)
# 
# # Calculate the proportion of each genre
# genre_prop <- genre_counts / total_movies
# 
# 
# kable(genre_counts, caption = "Number of movies within each genre (most movies appear in more than one genre)")


# Create a data.table with counts and proportions
#genre_summary <- data.table(
#  Genre = names(genre_counts),
#  Count = genre_counts,
#  Proportion = genre_prop
#)
#genre_summary <- genre_summary[order(genre_summary$Count, decreasing = TRUE), ]

#We can see that there is quite a difference in how many movies fall into each genre - the IMAX has only 29 movies listed, and one movie has no genre associated with it at all.

####### Plots ##############


# Plot times movie was rated vs the average rating for the movie
# edx_wide %>% group_by(movieId) %>%
#   summarise(n_times_rated = n(), avg_rating = mean(rating)) %>%
#   ggplot(aes(x = n_times_rated, y = avg_rating)) +
#   geom_point() +
#   geom_smooth() +
#   scale_x_log10() +
#   labs(x = "Times movie has been rated", y = "Mean rating of movie", title = "Effect of number of ratings on mean rating of movies")
# Movies appear to have higher ratings the more times they have been rated - which makes sense, as more people watch popular (highly rated) movies.

# Plot release year vs the average rating for each movie
# edx_wide %>% group_by(release_year) %>%
#   summarise(avg_rating = mean(rating)) %>%
#   ggplot(aes(release_year, avg_rating)) +
#   geom_point() +
#   geom_smooth() +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))
# There is a slight decrease in rating the more recent the movie is - is this because more recent movies have fewer ratings?

# Plot release_year vs times a movie has been rated
# edx_wide %>% group_by(release_year) %>%
#   summarise(n_times_rated = n()) %>%
#   ggplot(aes(x = release_year, y = n_times_rated)) +
#   geom_point() +
#   geom_smooth() +
#   scale_x_log10() +
#   labs(x = "Release year", y = "Number of times rated", title = "Effect of number of ratings on mean rating of movies")

# Plot userId against mean rating
# edx_wide %>% group_by(userId) %>% summarize(mean_rating = mean(rating)) %>%
#   ggplot(aes(mean_rating)) +
#   geom_histogram(fill = "sienna2") +
#   labs(x = "Mean rating", y = "Number of users", title = "Average user ratings")

# Plot movieId against mean rating
# edx_wide %>% group_by(movieId) %>% summarize(mean_rating = mean(rating)) %>%
#   ggplot(aes(mean_rating)) +
#   geom_histogram(fill = "palegreen3") +
#   labs(x = "Mean rating", y = "Number of movies", title = "Average movie ratings")


### These plots are made on original genres column #####
# edx %>% group_by(genres) %>%
#   summarize(mean_rating = mean(rating), n_rated = n()) %>%
#   ggplot(aes(x = n_rated, y = mean_rating, label = genres, colour = genres)) +
#   geom_point() +
#   geom_text_repel() +
#   labs(x = "Number of ratings", y = "Mean rating", title = "Average movie rating for different genres") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") # Rotate x-axis labels for better readability

# Plot genres against average rating
# edx %>% group_by(genres) %>% summarize(mean_rating = mean(rating)) %>% arrange(genres, by = mean_rating) %>%
#   ggplot(aes(x = genres, y = mean_rating, colour = genres)) +
#            geom_point() +
#            labs(x = "Number of ratings", y = "Mean rating", title = "Average movie rating for different genres") +
#            theme(axis.text.x = element_text(angle = 90, hjust = 1), legend.position = "none")


####Time effect plots
# Plot mean rating against time since release
# edx_use %>% group_by(movieId, years_since_release) %>%
#   summarize(mean_ratings = mean(rating), years_since_release = years_since_release) %>%
#   ggplot(aes(x = years_since_release, y = mean_ratings)) +
#   geom_point() +
#   labs(x = "Years since release", y = "Mean rating", title = "Mean rating based on years since movie was released")

# time_effect_summary <- edx %>% group_by(movieId) %>%
#   summarize(n_ratings = n(),
#             years = 2018 - first(rating_year),
#             title = title[1],
#             avg_rating = mean(rating)) %>%
#   mutate(ratings_per_year = n_ratings/years) %>%
#   group_by(ratings_per_year) %>%
#   summarize(avg_rating = mean(avg_rating))

# time_effect_summary %>% ggplot(aes(ratings_per_year, avg_rating)) +
#   geom_point() +
#   geom_smooth(method = lm, se = FALSE)

############## PREPARE DATA FOR MODEL TESTING ######################

######## FULL EDX TIDY DATASET (genres not split) - Train and test set creation ########

# Create train and test sets from edx - use same methodology as used for creation of edx set above
set.seed(1)

edx_test_index <- createDataPartition(y = edx_tidy$rating, times = 1, p = 0.1, list = FALSE)

edx_train <- edx_tidy[-edx_test_index,]

temp <- edx_tidy[edx_test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx_train set (to improve predictions as same movies are in both sets?)

edx_test <- temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx_test set back into edx_train set

removed <- anti_join(temp, edx_test)

edx_train <- rbind(edx_train, removed)

rm(edx_test_index, temp, removed) # Remove as not needed

setDT(edx_train) # Covert to data.table format if not already
setDT(edx_test) # Covert to data.table format if not already

######## FULL EDX WIDE DATASET (genres split) - Train and test set preparation of full dataset ########

# Create train and test sets from edx - use same methodology as used for creation of edx set above
set.seed(1)

edx_test_index_wide <- createDataPartition(y = edx_wide$rating, times = 1, p = 0.1, list = FALSE)

edx_train_wide <- edx_wide[-edx_test_index_wide,]

temp <- edx_wide[edx_test_index_wide,]

# Make sure userId and movieId in final hold-out test set are also in edx_train set (to improve predictions as same movies are in both sets?)

edx_test_wide <- temp %>%
  semi_join(edx_train_wide, by = "movieId") %>%
  semi_join(edx_train_wide, by = "userId")

# Add rows removed from edx_test set back into edx_train set

removed <- anti_join(temp, edx_test_wide)

edx_train_wide <- rbind(edx_train_wide, removed)

rm(edx_test_index_wide, temp, removed) # Remove as not needed

setDT(edx_train_wide) # Covert to data.table format if not already
setDT(edx_test_wide) # Covert to data.table format if not already


######### Scale the data (including final holdout test set) to make the models we create less affected by outliers and prevent features with larger scales dominating those with smaller scales.
# Set up method for scaling

preprocess_method <- function(train_data, test_data) {
  preprocess_options <- preProcess(train_data, method = c("center", "scale"))
  train_scaled <- predict(preprocess_options, train_data)
  test_scaled <- predict(preprocess_options, test_data)
  return(list(train = train_scaled, test = test_scaled, preProc = preprocess_options))
}

# Exclude the target variable 'rating' and other non-numeric features (for tidy dataset - if using wide, remove "-genres")

train_predictors <- edx_train %>% select(-rating, -genres)
test_predictors <- edx_test %>% select(-rating, -genres)

# Apply the preprocessing function

preprocessed_data <- preprocess_method(train_predictors, test_predictors)

edx_train_scaled <- preprocessed_data$train
edx_test_scaled <- preprocessed_data$test

# Add the target variable back to the scaled data

edx_train_scaled$rating <- edx_train$rating
edx_test_scaled$rating <- edx_test$rating

setDT(edx_train_scaled) # Covert to data.table format if not already
setDT(edx_test_scaled) # Covert to data.table format if not already

######## For final holdout dataset - remember to apply scaling if applicable
# preprocessed_data <- preprocess_method(edx_tidy, final_holdout_test)
# edx_scaled <- preprocessed_data$train
# final_holdout_test_scaled <- preprocessed_data$test
# preProc <- preprocessed_data$preProc


# Apply scaling if applicable

################### SMALLER DATASET FROM WIDE SET FOR FEATURE SELECTION ##################
# set.seed(1)
# edx_smaller_wide <- edx_wide %>%
#   group_by(rating) %>%
#   sample_n(size = round(50000 * n() / nrow(edx_wide))) %>%
#   ungroup()
# 
# # Create train and test set from smaller set
# set.seed(1)
# test_index_wide <- createDataPartition(y = edx_smaller_wide$rating, times = 1, p = 0.1, list = FALSE)
# 
# edx_smaller_train_wide <- edx_smaller_wide[-test_index_wide, ]
# 
# temp <- edx_smaller_wide[test_index_wide, ]
# 
# # Make sure userId and movieId in test set are also in train set
# edx_smaller_test_wide <- temp %>%
#   semi_join(edx_smaller_train_wide, by = "movieId") %>%
#   semi_join(edx_smaller_train_wide, by = "userId")
# 
# # Add rows removed from edx_test set back into edx_train set
# 
# removed <- anti_join(temp, edx_smaller_test_wide)
# 
# edx_smaller_train_wide <- rbind(edx_smaller_train_wide, removed)
# 
# # Verify the sampling
# 
# table(edx_smaller_train_wide$rating) # Check distribution in train set
# 
# table(edx_smaller_test_wide$rating)  # Check distribution in test set
# 
# # Clean up temporary variables
# 
# rm(temp, removed, test_index_wide)
# 
# setDT(edx_smaller_train_wide)
# setDT(edx_smaller_test_wide)


# Perform feature selection on the sample
# Example with LASSO
# library(glmnet)
# 
# # Prepare data for LASSO model
# x_sample <- model.matrix(rating ~ ., edx_small_train_wide)[, -1]
# y_sample <- edx_small_train_wide$rating
# 
# # Fit LASSO model
# lasso_sample_model <- cv.glmnet(x_sample, y_sample, alpha = 1)
# 
# # Get the coefficients of the best lambda
# best_lambda <- lasso_sample_model$lambda.min
# lasso_sample_coefficients <- coef(lasso_sample_model, s = lasso_sample_model$lambda.min)
# lasso_sample_coefficients[lasso_sample_coefficients != 0]
# 
# # Convert the coefficients to a data frame for easy viewing
# coefficients_df <- as.data.frame(as.matrix(lasso_sample_coefficients))
# coefficients_df$feature <- rownames(coefficients_df)
# colnames(coefficients_df)[1] <- "coefficient"
# 
# # Filter non-zero coefficients
# non_zero_coefficients <- coefficients_df[coefficients_df$coefficient != 0, ]
# zero_coefficients <- coefficients_df[coefficients_df$coefficient == 0, ]
# # Print non-zero coefficients and their corresponding features
# print(non_zero_coefficients)
# print(zero_coefficients)
# 
# #### Try random forest ###############
# 
# # Prepare data
# # Select predictors and target variable for training set
# x_train <- edx_smaller_train_wide[, !names(edx_smaller_train_wide) %in% "rating", with = FALSE]  # Exclude 'rating' column
# y_train <- edx_smaller_train_wide$rating  # Target variable
# 
# # Select predictors and target variable for testing set
# x_test <- edx_smaller_test_wide[, !names(edx_smaller_test_wide) %in% "rating", with = FALSE]  # Exclude 'rating' column
# y_test <- edx_smaller_test_wide$rating  # Target variable
# 
# 
# # Train the Random Forest model
# start_time <- Sys.time()
# 
# rf_model <- randomForest(y_train ~ ., data = x_train)
# 
# end_time <- Sys.time()
# elapsed_time <- end_time - start_time
# print(elapsed_time)
# 
# # Predict on the testing data
# predictions <- predict(rf_model, newdata = x_test)
# 
# # Evaluate model performance
# # For regression, you can use RMSE or MAE
# rmse <- sqrt(mean((predictions - y_test)^2))
# mae <- mean(abs(predictions - y_test))
# 
# # Print the evaluation metrics
# print(paste("RMSE:", rmse))
# print(paste("MAE:", mae))
# 
# # Get feature importance
# importance <- importance(rf_model)
# sorted_importance <- importance[order(importance[, 1], decreasing = TRUE), ]
# print(as.data.frame(sorted_importance))

######### Try recursive elimination ######## 

#Use x_train and y_train from the random forest model above

# # Define the model to use for RFE
# start_time <- Sys.time()
# 
# model <- train(y_train ~ ., data = x_train, method = "rf")
# 
# end_time <- Sys.time()
# elapsed_time <- end_time - start_time
# print(elapsed_time)
# 
# # Perform recursive feature elimination
# rfe_result <- rfe(x = x_train, y = y_train, sizes = c(1:ncol(x_train)), rfeControl = rfeControl(functions = rfFuncs), method = "rf")
# 
# # Print the results
# print(rfe_result)


######### SMALLER DATASET FROM TIDY SET FOR TESTING  - Create smaller set for testing models #############

# # Smaller sample set for testing code using stratified sampling
# set.seed(1)
# edx_smaller <- edx_tidy %>%
#   group_by(rating) %>%
#   sample_n(size = round(50000 * n() / nrow(edx_wide))) %>%
#   ungroup()
# 
# # Remove unnecessary columns
# #edx_smaller <- edx_smaller %>% select(-c(("no genres listed)")))
# 
# # Create train and test set from smaller set
# set.seed(1)
# test_index <- createDataPartition(y = edx_smaller$rating, times = 1, p = 0.1, list = FALSE)
# 
# edx_smaller_train <- edx_smaller[-test_index, ]
# 
# temp <- edx_smaller[test_index, ]
# 
# # Make sure userId and movieId in test set are also in train set
# edx_smaller_test <- temp %>%
#   semi_join(edx_smaller_train, by = "movieId") %>%
#   semi_join(edx_smaller_train, by = "userId")
# 
# # Add rows removed from edx_test set back into edx_train set
# 
# removed <- anti_join(temp, edx_smaller_test)
# 
# edx_smaller_train <- rbind(edx_smaller_train, removed)
# 
# # Verify the sampling
# 
# table(edx_smaller_train$rating) # Check distribution in train set
# 
# table(edx_smaller_test$rating)  # Check distribution in test set
# 
# # Clean up temporary variables
# 
# rm(temp, removed, test_index)

# # Scale if required
# set.seed(1)
# 
# # Exclude the target variable 'rating' and other non-numeric features if any
# train_predictors <- edx_smaller_train %>% select(-rating, -genres)
# test_predictors <- edx_smaller_test %>% select(-rating, -genres)
# 
# # Apply the preprocessing function
# preprocessed_data <- preprocess_method(train_predictors, test_predictors)
# edx_small_train_scaled <- preprocessed_data$train
# edx_small_test_scaled <- preprocessed_data$test
# 
# # Add the target variable back to the scaled data
# edx_small_train_scaled$rating <- edx_smaller_train$rating
# edx_small_test_scaled$rating <- edx_smaller_test$rating
# 
# 
# preprocessed_smaller <- preprocess_method(edx_smaller_train, edx_smaller_test)
# edx_small_train_scaled <- preprocessed_smaller$train
# edx_small_test_scaled <- preprocessed_smaller$test
# 
# rm(preprocessed_smaller)

#################### Allow parallel processing #############
# library(doParallel)
# cl <- makeCluster(detectCores() - 1)  # Use one less than the number of available cores
# registerDoParallel(cl)



########### Regularization ###############

# # Cross-validation to find best lambda values for regularization of userId, movieId and genres variables
#
# compute_rmse <- function(train_data, test_data, lambda_movie, lambda_user, lambda_genres) {
#   overall_mean_rating <- mean(edx_smaller_train$rating)
#
#   # Compute regularized movie effects
#   movie_effects <- edx_train %>%
#     group_by(movieId) %>%
#     summarize(b_i = sum(rating - overall_mean_rating) / (n() + lambda_movie), .groups = 'drop')
#
#   # Compute regularized user effects
#   user_effects <- edx_smaller_train %>%
#     left_join(movie_effects, by = "movieId") %>%
#     group_by(userId) %>%
#     summarize(b_u = sum(rating - overall_mean_rating - b_i) / (n() + lambda_user), .groups = 'drop')
#
#   # Compute regularized year effect
#   genre_effects <- edx_train %>%
#     left_join(movie_effects, by="movieId") %>%
#     left_join(user_effects, by="userId") %>%
#     group_by(genres) %>%
#     summarize(b_g = sum(rating - overall_mean_rating - b_i - b_u) / (n() + lambda_genres))
#
#   # Add regularized effects to the test data
#   edx_smaller_test <- edx_smaller_train %>%
#     left_join(movie_effects, by = "movieId") %>%
#     left_join(user_effects, by = "userId") %>%
#     left_join(genre_effects, by = "genres") %>%
#     mutate(b_i = ifelse(is.na(b_i), 0, b_i),
#            b_u = ifelse(is.na(b_u), 0, b_u),
#            b_y = ifelse(is.na(b_g), 0, b_g))
#
#   # Predict ratings
#   predicted_ratings <- overall_mean_rating + edx_smaller_test$b_i + edx_smaller_test$b_u + edx_smaller_test$b_y
#
#   # Calculate RMSE
#   rmse <- sqrt(mean((edx_smaller_test$rating - predicted_ratings)^2))
#   return(rmse)
# }
#
#
# # Set up cross-validation
# k <- 5
# folds <- createFolds(edx_smaller_train$rating, k = k)
#
# # Define a grid of lambda values to try
# lambda_values <- seq(0.1, 5, 0.5)
#
# # Initialize a data frame to store results
# results <- expand.grid(lambda_movie = lambda_values, lambda_user = lambda_values, lambda_year = lambda_values)
# results$rmse <- NA
#
# # Perform cross-validation
# for (i in 1:nrow(results)) {
#   lambda_movie <- results$lambda_movie[i]
#   lambda_user <- results$lambda_user[i]
#   lambda_year <- results$lambda_year[i]
#
#   rmse_values <- c()
#
#   for (fold in folds) {
#     train_fold <- edx_smaller_train[-fold, ]
#     test_fold <- edx_smaller_train[fold, ]
#
#     rmse <- compute_rmse(train_fold, test_fold, lambda_movie, lambda_user, lambda_year)
#     rmse_values <- c(rmse_values, rmse)
#   }
#
#   results$rmse[i] <- mean(rmse_values)
# }
#
# # Find the best lambda values
# best_params <- results[which.min(results$rmse), ]
# best_lambda_movie <- best_params$lambda_movie
# best_lambda_user <- best_params$lambda_user
# best_lambda_year <- best_params$lambda_year
# best_lambda_genre <- best_params$lambda_genre
#
# print(best_params) #lambda_user and lamba_movie are both 0.1. RMSE for this is 0.6272254, higher than the 0.2 something I got with 10 000 dataset.


############## Full dataset run of regularization model with best lambda found above #################

# Apply best lambda to the full dataset
best_lambda_movie <- 0.1
best_lambda_user <- 0.1
best_lambda_time <- 0.1
best_lambda_genre <- 0.1

###### Individually calculating the effects ######

#Calculate overall mean rating
overall_mean_rating <- mean(edx_train$rating)

# Calculate movieId effect with the best lambda
movie_effects <- edx_train%>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - overall_mean_rating) / (n() + best_lambda_movie))

# Calculate movie_avg effect with the best lambda
movie_avg_effects <- edx_train%>%
  group_by(movie_avg_rating) %>%
  summarize(b_i_a = sum(rating - overall_mean_rating) / (n() + best_lambda_movie))

# Calculate userId effect with the best lambda
user_effects <- edx_train %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - overall_mean_rating) / (n() + best_lambda_user))

user_avg_effects <- edx_train %>%
  group_by(user_avg_rating) %>%
  summarize(b_u_a = sum(rating - overall_mean_rating) / (n() + best_lambda_user))

# Calculate timestamp effects with regularization
# Calculate global time effects using the entire dataset
global_time_effects <- edx_tidy %>%
  group_by(timestamp) %>%
  summarize(b_t = sum(rating - overall_mean_rating - b_i - b_u - b_i_a - b_u_a) / (n() + best_lambda_time)) %>%
  ungroup()

# Check for NAs in global_time_effects
any(is.na(global_time_effects$b_t))


time_effects <- edx_train %>%
  group_by(timestamp) %>%
  summarize(b_t = sum(rating - overall_mean_rating) / (n() + best_lambda_time))

# Calculate genre effects with regularization - with all genres separated
# genre_effects <- edx_train %>%
#   pivot_longer(cols = starts_with("genre"), names_to = "genre", values_to = "value") %>%
#   filter(value == 1) %>%
#   group_by(genre) %>%
#   summarize(b_g = sum(rating - overall_mean_rating) / (n() + best_lambda_genre))

# # Calculate genres effect - with genres column intact
# genre_effects <- edx_train %>%
#   group_by(genres) %>%
#   summarize(b_g = sum(rating - overall_mean_rating) / (n() + best_lambda_genre))

setDT(user_effects)
setDT(user_avg_effects)
setDT(movie_effects)
setDT(movie_avg_effects)
setDT(time_effects)
# setDT(genre_effects)



# ###### Predictions with individual effects #######

predicted_ratings_reg <- edx_test %>%
  left_join(movie_effects, by="movieId") %>%
  left_join(movie_avg_effects, by="movie_avg_rating") %>%
  left_join(user_effects, by="userId") %>%
  left_join(user_avg_effects, by="user_avg_rating") %>%
  #left_join(time_effects, by="timestamp") %>%
  mutate(pred = overall_mean_rating + b_i + b_i_a + b_u + b_u_a) %>%
  pull(pred)

# Joining movie effects
edx_test_joined <- edx_test %>%
  left_join(movie_effects, by="movieId") %>%
  left_join(movie_avg_effects, by="movie_avg_rating") %>%
  left_join(user_effects, by="userId") %>%
  left_join(user_avg_effects, by="user_avg_rating") %>%
  left_join(time_effects, by="timestamp")

# Identify rows with NAs
na_rows <- edx_test_joined[is.na(edx_test_joined$pred), ]
print(na_rows)

#rowwise() %>%
#mutate(b_g_sum = sum(c_across(starts_with("genre_")) * genre_effects$b_g[match(names(c_across(starts_with("genre_"))), genre_effects$genre)], na.rm = TRUE)) %>%

# # Predictions movieId only
# predicted_ratings_reg_movie <- edx_test %>%
#   left_join(movie_effects, by="movieId") %>%
#   mutate(pred = overall_mean_rating + b_i) %>%
#   pull(pred)
#
# Predictions movie_avg only
# predicted_ratings_reg_movie <- edx_test %>%
#   left_join(movie_avg_effects, by="movie_avg_rating") %>%
#   mutate(pred = overall_mean_rating + b_i) %>%
#   pull(pred)
#
#
# # Predictions userId only
# predicted_ratings_reg_user <- edx_test %>%
#   left_join(user_effects, by="userId") %>%
#   mutate(pred = overall_mean_rating + b_u) %>%
#   pull(pred)
# 
# # Predictions release_year only
# predicted_ratings_reg_year <- edx_test %>%
#   left_join(year_effects, by="release_year") %>%
#   mutate(pred = overall_mean_rating + b_y) %>%
#   pull(pred)
# 
# # Predictions genres only
# predicted_ratings_reg_genres <- edx_test %>%
#   left_join(genre_effects, by="genres") %>%
#   mutate(pred = overall_mean_rating + b_g) %>%
#   pull(pred)
# 
# # Predictions genres only - with genres column separated
# # predicted_ratings_reg_genres <- edx_test %>%
# #   rowwise() %>%
# #   mutate(b_g_sum = sum(c_across(starts_with("genre_")) * genre_effects$b_g[match(names(c_across(starts_with("genre_"))), genre_effects$genre)], na.rm = TRUE)) %>%
# #   ungroup() %>%
# #   mutate(pred = overall_mean_rating + b_g_sum) %>%
# #   pull(pred)
# 
# # Predictions userId and movieId only
# predicted_ratings_reg_i_u <- edx_test %>%
#   left_join(movie_effects, by="movieId") %>%
#   left_join(user_effects, by="userId") %>%
#   mutate(pred = overall_mean_rating + b_i + b_u) %>%
#   pull(pred)
# 
# # Predictions userId, movieId and genres only
# predicted_ratings_reg_i_u_g <- edx_test %>%
#   left_join(movie_effects, by="movieId") %>%
#   left_join(user_effects, by="userId") %>%
#   left_join(genre_effects, by="genres") %>%
#   mutate(pred = overall_mean_rating + b_i + b_u + b_g) %>%
#   pull(pred)

##### Bayesian average

# Compute global mean rating
global_mean <- mean(edx_train$rating)

# Define prior parameters
m_user <- 20
m_movie <- 20

# Compute Bayesian user effect
user_effects_bayes <- edx_train %>%
  group_by(userId) %>%
  summarize(user_avg_rating = mean(rating),
            user_count = n()) %>%
  mutate(b_u_bayes = (user_count * user_avg_rating + m_user * global_mean) / (user_count + m_user)) %>%
  select(userId, b_u_bayes)

# Compute Bayesian movie effect
movie_effects_bayes <- edx_train %>%
  group_by(movieId) %>%
  summarize(movie_avg_rating = mean(rating),
            movie_count = n()) %>%
  mutate(b_i_bayes = (movie_count * movie_avg_rating + m_movie * global_mean) / (movie_count + m_movie)) %>%
  select(movieId, b_i_bayes)

# Apply Bayesian effects to test data
predicted_ratings_bayes <- edx_test %>%
  left_join(movie_effects_bayes, by = "movieId") %>%
  left_join(user_effects_bayes, by = "userId") %>%
  mutate(pred = global_mean + b_i_bayes + b_u_bayes) %>%
  pull(pred)

# Evaluate model performance
rmse_bayes <- sqrt(mean((predicted_ratings_bayes - edx_test$rating)^2))
print(rmse_bayes)

# ###### RMSEs #####
# 
# # Set up RMSE function
calculate_rmse <- function(predictions, true_values) {
  sqrt(mean((predictions - true_values)^2))
}

# Write function for RMSE calculation
# RMSE <- function(true_ratings, predicted_ratings){
#  sqrt(mean((true_ratings - predicted_ratings)^2))}
 
# # RMSE userId only
# userId_reg_rmse <- calculate_rmse(predicted_ratings_reg_user, edx_test$rating)
# 
# # RMSE movieId
# movieId_reg_rmse <- calculate_rmse(predicted_ratings_reg_movie, edx_test$rating)

# # RMSE movie_avg
# movieId_avg_reg_rmse <- calculate_rmse(predicted_ratings_reg_movie, edx_test$rating)

# # RMSE year only
# release_year_reg_rmse <- calculate_rmse(predicted_ratings_reg_year, edx_test$rating)
# 
# # RMSE genres only
# genres_reg_rmse <- calculate_rmse(predicted_ratings_reg_genres, edx_test$rating)
# 
# # RMSE all together
all_reg_rmse <- calculate_rmse(predicted_ratings_reg, edx_test$rating)
all_reg_rmse
 
# # RMSE userId and movieId only - lowest RMSE 0.8843305
# user_movie_reg_rmse <- calculate_rmse(predicted_ratings_reg_i_u, edx_test$rating)
# 
# # RMSE userId, movieId and genres - adding genres increases RMSE somewhat 0.9440207
# user_movie_genre_reg_rmse <- calculate_rmse(predicted_ratings_reg_i_u_g, edx_test$rating)
# 
# rmse_results <- tibble(method = c("UserId effect", "MovieId effect", "Release year effect", "Genres effect", "User, movie, release year and genres effect", "User and movie effect", "User, movie and genres effect"), RMSE = userId_reg_rmse, movieId_reg_rmse, release_year_reg_rmse, genres_reg_rmse, user_movie_year_genre_reg_rmse, user_movie_reg_rmse, user_movie_genre_reg_rmse)
# rmse_results

# ################ k-Nearest Neighbours models ###############
#
#set.seed(1)

# knn model - all predictors
fit_knn_all <- train(rating ~ timestamp + movie_avg_rating + user_avg_rating + user_sd_rating,
                       method = "knn",
                       data = edx_train_scaled,
                       tuneGrid = data.frame(k = seq(1, 100, 5)),
                       #trControl = trainControl(method = "cv", number = 5)) #Cross validation
)
fit_knn_all$bestTune

pred_knn_all <- predict(fit_knn_all, edx_small_test_scaled)

# RMSE kNN
#RMSE(edx_small_test_scaled$rating, pred_knn_all)
#RMSE(edx_small_test_scaled$rating, pred_knn_user_movie) # 0.9948127 for small set
#RMSE(edx_small_test_scaled$rating, pred_knn_genres) #  for small set


# # knn model - userId and movieId
# 
# fit_knn <- train(rating ~ userId + movieId,
#                      method = "knn",
#                      data = edx_small_train_scaled,
#                      tuneGrid = data.frame(k = seq(1, 100, 2)),
#                      trControl = trainControl(method = "cv", number = 5)) #Cross validation
# fit_knn_user_movie$bestTune
# 
# pred_knn_user_movie <- predict(fit_knn_user_movie, edx_small_test_scaled)
# 
# 
# # knn model - genres
# fit_knn_genres <- train(rating ~ genres,
#                      method = "knn",
#                      data = edx_small_train_scaled,
#                      tuneGrid = data.frame(k = seq(1, 100, 2)),
#                      trControl = trainControl(method = "cv", number = 5)) #Cross validation
# fit_knn_genres$bestTune
# 
# pred_knn_genres <- predict(fit_knn_genres, edx_small_test_scaled)
# 
# # knn model - userId, movieId and genres
# 
# fit_knn_user_movie_genres <- train(rating ~ userId + movieId + genres,
#                      method = "knn",
#                      data = edx_small_train_scaled,
#                      tuneGrid = data.frame(k = seq(1, 100, 2)),
#                      trControl = trainControl(method = "cv", number = 5)) #Cross validation
# fit_knn_user_movie_genres$bestTune
# 
# pred_knn_user_movie_genres <- predict(fit_knn_user_movie_genres, edx_small_test_scaled)

#
# ################ k k-Nearest Neighbours models ###############
# #k k-NN model
#
# kknn_model <- kknn(rating ~ ., train = edx_small_train_scaled, test = edx_small_test_scaled, k = 10, distance = 2, kernel = "optimal")
#
# pred_k_knn_all <- fitted(kknn_model)
#
# # Define a function to calculate RMSE for a given k
#  calculate_rmse <- function(k, train_data, test_data) {
#    kknn_model <- kknn(rating ~ ., train = train_data, test = test_data, k = k, distance = 3, kernel = "optimal")
#    predictions <- fitted(kknn_model)
#    rmse <- sqrt(mean((predictions - test_data$rating)^2))
#    return(rmse)
#  }
#
# # Use sapply to find the best k
# k_values <- seq(200, 300, 2)
# rmse_values <- sapply(k_values, calculate_rmse, train_data = edx_small_train_scaled, test_data = edx_small_test_scaled)
#
# best_k <- k_values[which.min(rmse_values)]
# best_rmse <- min(rmse_values)
#
# cat("Best k:", best_k, "with RMSE:", best_rmse, "\n")
#
# ####### Random forest models ###########
#rf model:
fit_rf_most_imp <- train(rating ~ movie_avg_rating + user_avg_rating + user_sd_rating,
                     method = "rf",
                     data = edx_train,
                     tuneGrid = data.frame(mtry = 4),
                     #trControl = trainControl(method = "cv", number = 5), #Cross validation
                     importance = TRUE,
                     ntree = 100)


 fit_rf_all$bestTune
varImp(fit_rf_all)
pred_rf_all <- predict(fit_rf_all, edx_small_test_scaled)
#
# fit_rf_user_movie <- train(rating ~ userId + movieId,
#                            method = "rf",
#                            data = edx_small_train_scaled,
#                            tuneGrid = expand.grid(mtry = seq(2, 8, 2)),
#                            trControl = trainControl(method = "cv", number = 5), #Cross validation
#                            importance = TRUE,
#                            ntree = 300)
# 
# fit_rf_user_movie$bestTune
# varImp(fit_rf_user_movie)
# pred_rf_user_movie <- predict(fit_rf_user_movie, edx_small_test_scaled)


# fit_rf_genres <- train(rating ~ Comedy + Romance + Action + Crime + Thriller + Drama + Adventure + Children + Fantasy + War + Animation + Musical + Horror + Documentary + IMAX,
#                     method = "rf",
#                     data = edx_small_train_scaled,
#                     tuneGrid = data.frame(mtry = 2:5),
#                     trControl = trainControl(method = "cv", number = 5)) #Cross validation
#
# fit_rf_genres$bestTune
#
# pred_rf_genres <- predict(fit_rf_genres, edx_small_test_scaled)

#### Regression models for userId and movieId

### biglm package - same result as speedglm abut slower
# library(biglm)
# 
# # Fit the model
# model_biglm <- biglm(rating ~ userId + movieId + timestamp + movie_avg_rating + user_avg_rating + movie_sd_rating + user_sd_rating + user_rating_count + movie_rating_count, data = edx_train_scaled)
# 
# # Predict on test set
# predicted_ratings_biglm <- predict(model_biglm, edx_test_scaled)
# 
# # Calculate RMSE
# rmse_biglm <- RMSE(edx_test_scaled$rating, predicted_ratings_biglm)
# print(rmse_biglm) # 1.059393 on full scaled dataset, 0.8693229 after adding new and most important predictors

### speedglm package - keep for ensemble
library(speedglm)

# Fit the model
model_speedglm <- speedglm(rating ~ userId + movieId + timestamp + movie_avg_rating + user_avg_rating + movie_sd_rating + user_sd_rating + user_rating_count + movie_rating_count, data = edx_train_scaled)

# Predict on test set
predicted_ratings_speedglm <- predict(model_speedglm, edx_test_scaled)

# Calculate RMSE
rmse_speedglm <- RMSE(edx_test_scaled$rating, predicted_ratings_speedglm)
print(rmse_speedglm) # 1.060032 on full scaled dataset, 0.8693229 with new and most important predictors

### glmnet package - offers regularisation - use scaled data for this
#
#library(glmnet)
#
# Prepare data for glmnet
# x_train <- model.matrix(rating ~ userId + movieId, data = edx_train)[, -1]
# y_train <- edx_train$rating
# x_test <- model.matrix(rating ~ userId + movieId, data = edx_test)[, -1]
# 
# # Fit the model
# model_glmnet <- cv.glmnet(x_train, y_train, alpha = 0)
# 
# # Predict on test set
# predicted_ratings_glmnet <- predict(model_glmnet, s = "lambda.min", newx = x_test)
# 
# # Calculate RMSE
# rmse_glmnet <- RMSE(edx_test$rating, predicted_ratings_glmnet)
# print(rmse_glmnet)


### xgboost package - offers regularisation - do not use scaled data = keep for ensemble
library(xgboost)

# Prepare data for xgboost
dtrain <- xgb.DMatrix(data = as.matrix(edx_train[, c("userId", "movieId", "timestamp", "movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "user_rating_count", "movie_rating_count")]), label = edx_train$rating)
dtest <- xgb.DMatrix(data = as.matrix(edx_test[, c("userId", "movieId", "timestamp", "movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "user_rating_count", "movie_rating_count")]))

# Fit the model
params_reg <- list(objective = "reg:squarederror")
model_xgboost <- xgboost(data = dtrain, params = params, nrounds = 200)

# Predict on test set
predicted_ratings_xgboost <- predict(model_xgboost, dtest)

# Calculate RMSE
rmse_xgboost <- RMSE(edx_test$rating, predicted_ratings_xgboost)
print(rmse_xgboost) # 0.9849198 with big set, 0.8519437 with the new and most important predictors

### h20 package - scale data
# library(h2o)
# 
# # Initialize H2O
# h2o.init()
# 
# # Convert data to H2O frame
# h2o_train <- as.h2o(edx_train_scaled)
# h2o_test <- as.h2o(edx_test_scaled)
# 
# # Fit the model
# model_h2o <- h2o.glm(y = "rating", x = c("userId", "movieId", "timestamp", "movie_avg_rating", "user_avg_rating", "movie_sd_rating", "user_sd_rating", "user_rating_count", "movie_rating_count"), training_frame = h2o_train)
# 
# # Predict on test set
# predicted_ratings_h2o <- h2o.predict(model_h2o, h2o_test)$predict
# 
# # Calculate RMSE
# rmse_h2o <- RMSE(as.vector(edx_test_scaled$rating), as.vector(predicted_ratings_h2o))
# print(rmse_h2o) # 1.002342 with small scaled set, 1.060032 on large set
# 
# # Shutdown H2O
# h2o.shutdown(prompt = FALSE)


##### Ensemble models ########
#
# #Ensemble 1 - this is worse than individual
# ensemble_pred <- (pred_rf_all + pred_knn_all + pred_k_knn_all) / 3
#
# Ensemble - speed and xgboost
pred_ensemble_speed_xgboost <- (predicted_ratings_xgboost + predicted_ratings_speedglm) / 2

RMSE(pred_ensemble_speed_xgboost, edx_test$rating) # .8570211 , slightly worse than xgboost alone

# Combine the predictions into a meta-dataset for training
meta_train <- data.frame(
  pred_speedglm = predicted_ratings_speedglm,
  pred_xgboost = predicted_ratings_xgboost,
  rating = edx_test_scaled$rating
)

setDT(meta_train)

# Split the meta-dataset into training and testing sets (here using a simple train/test split)
set.seed(1)
train_indices <- createDataPartition(meta_train$rating, p = 0.8, list = FALSE)
meta_train_set <- meta_train[train_indices,]
meta_test_set <- meta_train[-train_indices,]

# Train the meta-model using linear regression
meta_model <- lm(rating ~ pred_speedglm + pred_xgboost, data = meta_train_set)

# Generate predictions on the meta-test set
meta_predictions <- predict(meta_model, meta_test_set)

# Calculate RMSE for the stacked model
meta_rmse <- RMSE(meta_test_set$rating, meta_predictions)
print(meta_rmse)

# Perform cross-validation on the meta-model
folds <- createFolds(meta_train$rating, k = 5)
cv_rmse <- vector()

for(i in 1:5) {
  fold_train <- meta_train[folds[[i]],]
  fold_val <- meta_train[-folds[[i]],]
  
  meta_model <- lm(rating ~ pred_speedglm + pred_xgboost, data = fold_train)
  meta_predictions <- predict(meta_model, fold_val)
  
  fold_rmse <- RMSE(fold_val$rating, meta_predictions)
  cv_rmse <- c(cv_rmse, fold_rmse)
}

# Calculate average RMSE across folds
mean_cv_rmse <- mean(cv_rmse)
print(mean_cv_rmse)

####### RMSE for model testing ######




# RMSE k_kNN
#RMSE(edx_small_test_scaled$rating, pred_k_knn_all)

# RMSE rf
#RMSE(edx_small_test_scaled$rating, pred_rf_all)
#RMSE(edx_small_test_scaled$rating, pred_rf_user_movie)
#RMSE(edx_small_test_scaled$rating, pred_rf_genres)

# RMSE ensemble_pred
# RMSE(edx_small_test_scaled$rating, ensemble_pred)

##### RMSE for final model test #####

#RMSE(final_holdout_test, pred_final_model)

######################End of script ##############################



#Check for overtraining and oversmoothing - look into visual representations for this as well as the accuracy. Compare to edx_test?



# #Set up the model like we did in the course
#
# #Write function for RMSE calculation
# RMSE <- function(true_ratings, predicted_ratings){
#   sqrt(mean((true_ratings - predicted_ratings)^2))}
#
# #Get the mean rating of all movies
# mean_ratings <- mean(as.numeric(edx_train$rating))
#
# #Get the RMSE based on mean only
# mean_only_rmse <- RMSE(edx_test$rating, mean_ratings)
# mean_only_rmse
#
# #Get the movie effect
# mean_per_movie <- edx_train %>%
#   group_by(movieId) %>%
#   summarize(b_i = mean(rating - mean_ratings))
#
# mean_per_movie
#
# #How does adding the movie effect affect the RMSE?
#
# predicted_ratings <- mean_ratings + edx_test %>%
#   left_join(mean_per_movie, by='movieId') %>%
#   pull(b_i)
#
# mean_and_movie_effects_rmse <- RMSE(predicted_ratings, edx_test$rating)
# mean_and_movie_effects_rmse
#
# #Get the user effect
# mean_per_user <- edx_train %>%
#   left_join(mean_per_movie, by='movieId') %>%
#   group_by(userId) %>%
#   summarize(b_u = mean(rating - mean_ratings - b_i))
#
# mean_per_user
#
# #How does adding user effect affect the RMSE?
#
# predicted_ratings <- edx_test %>%
#   left_join(mean_per_movie, by='movieId') %>%
#   left_join(mean_per_user, by='userId') %>%
#   mutate(pred = mean_ratings + b_i + b_u) %>%
#   pull(pred)
#
# mean_user_movie_rmse <- RMSE(predicted_ratings, edx_test$rating)
# mean_user_movie_rmse
#
#
# #RMSEs for each model
# rmse_results <- tibble(method = c("Just the average", "Movie effect", "Movie and user effects"), RMSE = c(mean_only_rmse, mean_and_movie_effects_rmse, mean_user_movie_rmse))
#
# rmse_results