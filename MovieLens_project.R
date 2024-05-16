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

library(lubridate)
library(dplyr)
library(ggrepel)

######### Tidy and pre-process data ###############

#Tidy timestamp into year only
edx <- edx %>% mutate(rating_year = year(as_datetime(timestamp)))

#Tidy data to separate release year from the title
edx <- edx %>% separate(title, c("title", "release_year"), -6) # Get the release year, but it's still in brackets
release_year_new <- str_extract(edx$release_year, "\\d{4}") # Remove brackets
edx$release_year <- release_year_new # Overwrite the column in main dataset with the one without brackets
rm(release_year_new) # Remove as no longer needed

#The genre column is more complex as it can have several entries per row - to be able to analyse this in a better way, we will create a "dummy" variable for each genre where the values are 1 (in genre) or 0 (not in genre), also know as "one-hot encoding". We know that one movie does not have any genres associated, so we will double check that this movie will only have "0" for every genre column.

edx <- edx %>% separate_rows(genres, sep = "\\|") %>% # Separate out each genre to a new row by the pipe marker
  mutate(genre_indicator = 1) %>% # Add 1 to the indicator column for each row
  pivot_wider(names_from = genres, values_from = genre_indicator, values_fill = 0) # Pivot the data so each genre has a column

############# Familiarise ourselves with data ##############

#Smaller sample set for testing code
# set.seed(1)
# edx_smaller_index <- sample(nrow(edx), size = 100000)
# edx_smaller <- edx[edx_smaller_index, ]

#Do some summary statistics and overview of data

#Plot times movie was rated vs the average rating for the movie
# edx %>% group_by(movieId) %>% 
#   summarise(n_times_rated = n(), avg_rating = mean(rating)) %>% 
#   ggplot(aes(avg_rating, n_times_rated)) + 
#   geom_point() + 
#   labs(x = "Mean rating of movie", y = "Times movie has been rated", title = "Effect of number of ratings on mean rating of movies")
#Movies appear to have higher ratings the more times they have been rated - which makes sense, as more people watch popular (highly rated) movies.

#Plot release year vs the average rating for each movie
# edx %>% group_by(release_year) %>% 
#   summarise(avg_rating = mean(rating)) %>% 
#   ggplot(aes(release_year, avg_rating)) +
#   geom_point() +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))
#There is a slight decrease in rating the more recent the movie is - is this because more recent movies have fewer ratings?

#Plot userId against mean rating
# edx %>% group_by(userId) %>% summarize(mean_rating = mean(rating)) %>%
#   ggplot(aes(mean_rating)) +
#   geom_histogram(fill = "sienna2") +
#   labs(x = "Mean rating", y = "Number of users", title = "Average user ratings")

#Plot movieId against mean rating
# edx %>% group_by(movieId) %>% summarize(mean_rating = mean(rating)) %>%
#   ggplot(aes(mean_rating)) +
#   geom_histogram(fill = "palegreen3") +
#   labs(x = "Mean rating", y = "Number of movies", title = "Average movie ratings")

#Plot genres against average rating
# genre_summary <- edx %>% separate_rows(genres, sep = "\\|") %>%
#   group_by(genres) %>%
#   summarise(mean_rating = mean(rating), n = n())

###These plots are made on original genres column #####
# ggplot(genre_summary, aes(y = mean_rating, x = n, label = genres, colour = genres)) +
#   geom_point() +
#   geom_text_repel() +
#   labs(y = "Mean rating", x = "Number of ratings", title = "Average movie rating for different genres") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") # Rotate x-axis labels for better readability

#Plot mean rating against time since release
# edx_smaller %>% group_by(movieId) %>% 
#   summarize(n_ratings = n(), year = first(rating_year)) %>% 
#   ggplot(aes(factor(year), n_ratings)) + 
#   geom_boxplot() + 
#   scale_y_sqrt() +
#   labs(x = "Year", y = "Number of ratings", title = "Number of ratings per movie per year")

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


############## Train and test set creation #################

#Scale the data (including final holdout test set) to make the models we create less affected by outliers and prevent features with larger scales dominating those with smaller scales.

preprocess_method <- function(data) {
  preprocess_options <- preProcess(data, method = c("center", "scale"))
  scaled_data <- predict(preprocess_options, data)
  return(scaled_data)}

edx_scaled <- preprocess_method(edx)
final_holdout_test_scaled <- preprocess_method(final_holdout_test)

#Create train and test sets from edx - use same methodology as used for creation of edx set above
edx_test_index <- createDataPartition(y = edx_scaled$rating, times = 1, p = 0.1, list = FALSE)

edx_train <- edx_scaled[-edx_test_index,]
temp <- edx_scaled[edx_test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx_train set (to improve predictions as same movies are in both sets?)
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx_test set back into edx_train set (so no data points are unused?)
removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(edx_test_index, temp, removed) #Remove as not needed

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
# #RMSEs for each model
# rmse_results <- tibble(method = c("Just the average", "Movie effect", "Movie and user effects"), RMSE = c(mean_only_rmse, mean_and_movie_effects_rmse, mean_user_movie_rmse))
# 
# rmse_results #But can't use this as it is code we used in the machine learning course


#Set up model in a different way to the course

set.seed(1)
k <- c(seq(1, 100, 5))
lambda <- c(0.001, 0.01, 0.1, 1, 10, 100)

#To tune the model further, we write a function that provides custom distance weighting within the knn model:

# distance_function <- function(distances) {
#   weights <- 1/distances
#   return(weights)
# }
# 
# fit_user_knn <- train(rating ~ userId ,
#                   method = "knn",
#                   data = edx_train,
#                   weights = distance_function, # Distance weighting
#                   tuneGrid = expand.grid(k = k, lambda = lambda),
#                   trControl = trainControl(method = "cv", number = 5)) #Cross validation
# 
# fit_movie_knn <- train(rating ~ movieId ,
#                   method = "knn",
#                   data = edx_train,
#                   tuneGrid = data.frame(k = seq(1, 100, 2)),
#                   trControl = trainControl(method = "cv", number = 5)) #Cross validation
# 
# fit_user_movie_knn <- train(rating ~ userId + movieId ,
#                   method = "knn",
#                   data = edx_train,
#                   tuneGrid = data.frame(k = seq(1, 100, 2)),
#                   trControl = trainControl(method = "cv", number = 5)) #Cross validation
# 
# 
# fit_genres_knn <- train(rating ~ userId + movieId ,
#                         method = "knn",
#                         data = edx_train,
#                         tuneGrid = data.frame(k = seq(1, 100, 2)),
#                         trControl = trainControl(method = "cv", number = 5)) #Cross validation






#Need a genre effect variable
#What does the genre column data look like?
# 
# edx_train %>% group_by(genres) %>% summarize(mean_per_genre = mean(rating), se = sd(rating)/sqrt(n())) %>%
#   mutate(genres = reorder(genres, mean_per_genre)) %>%
#      ggplot(aes(x = genres, y = mean_per_genre, ymin = mean_per_genre - 2*se, ymax = mean_per_genre + 2*se)) +
#      geom_point() +
#      geom_errorbar() +
#      theme(axis.text.x = element_text(angle = 90, hjust = 1))

#How to turn this into a usable variable for predictions?
#In the course the description was (Y_{u,i} = \mu + b_i + b_u + \sum_{k=1}^{K} x_{u,i}^{k} \beta_k + \varepsilon_{u,i}) with x_{u,i}^{k} = 1 if g_u,i is genre k



#Need a time effect variable




######################End of script to get the RMSE##############################
#Use regularization (penalised RMSE) at the end
#Do something like this code (from teh book):
#To select λ, we can use cross validation:
#   
# lambdas <- seq(0, 10, 0.1)
# 
# sums <- colSums(y - mu, na.rm = TRUE)
# rmses <- sapply(lambdas, function(lambda){
#   b_i <-  sums / (n + lambda)
#   fit_movies$b_i <- b_i
#   left_join(test_set, fit_movies, by = "movieId") |> mutate(pred = mu + b_i) |> 
#     summarize(rmse = RMSE(rating, pred)) |>
#     pull(rmse)
# })










#Can I do confusion matrix on this data? Is it going to be relevant?
#Prevalence - is this going to have a bearing in genres?

  
#Perform cross validation on the edx_train set - look into how to do this with bootstrap


#Check for overtraining and oversmoothing - look into visual representations for this as well as the accuracy. Compare to edx_test?


#Try different algorithms and decide on best one/best ensemble, use edx_test for this


#Do SVA and or PCA analysis on genres?

