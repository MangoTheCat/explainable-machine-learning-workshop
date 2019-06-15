
# dalex -------------------------------------------------------------------

library(DALEX)
library(randomForest)
library(ggplot2)

# train random forest and linear model
str(apartments)
set.seed(519)
apartments_rf_model <- randomForest::randomForest(m2.price ~ ., 
                                                  data = apartments)
predicted_rf <- predict(apartments_rf_model, apartments_test)

apartments_lm_model <-  lm(m2.price ~ ., data = apartments)
predicted_lm <- predict(apartments_lm_model, apartments_test)


# root mean square 
sqrt(mean((predicted_rf - apartments_test$m2.price)^2)) 
sqrt(mean((predicted_lm - apartments_test$m2.price)^2))


# create dalex explainer object
explainer_lm <- DALEX::explain(model = apartments_lm_model, 
                               data = apartments_test[,2:6], 
                               y = apartments_test$m2.price)
explainer_rf <- DALEX::explain(model = apartments_rf_model, 
                               data = apartments_test[,2:6], 
                               y = apartments_test$m2.price)

# inspect object
explainer_lm



# dalex: How good is the model? > Model performance audit -----------------

mp_lm <- model_performance(explainer_lm)
mp_rf <- model_performance(explainer_rf)

# plot absolute residuals
# via inverse cdf or boxplot
plot(mp_lm, mp_rf)
plot(mp_lm, mp_rf, geom = "boxplot")



# dalex: How good is the model? > Goodness of fit -------------------------

# mp object contains predicted, observed, residuals (= diff)
str(mp_rf)

ggplot(mp_rf, 
       aes(observed, diff)) +
  geom_point() + 
  xlab("Observed") + 
  ylab("Predicted - Observed") + 
  ggtitle("Diagnostic plot") + 
  theme_mi2()



# dalex: How does the model work? > All variables -------------------------

vi_rf <- variable_importance(explainer_rf, 
                             loss_function = loss_root_mean_square)
vi_lm <- variable_importance(explainer_lm, 
                             loss_function = loss_root_mean_square)

plot(vi_rf, vi_lm)



# dalex: How does the model work? > Continuous variable -------------------

# partial dependency plots
sv_rf  <- single_variable(explainer_rf, 
                          variable = "construction.year", 
                          type = "pdp")
sv_lm  <- single_variable(explainer_lm, 
                          variable = "construction.year", 
                          type = "pdp")
plot(sv_rf, sv_lm)


# dalex: How does the model work? > Categorical variable ------------------

svd_rf  <- single_variable(explainer_rf, variable = "district", type = "factor")
svd_lm  <- single_variable(explainer_lm, variable = "district", type = "factor")
plot(svd_rf, svd_lm)



# Understanding a single prediction: Variable attribution -----------------

# Identify the top outlier for RF model
max_diff_rf  <- which.max(abs(mp_rf$diff))

# See what the corresponding observed and predicted values are
mp_rf$predicted[max_diff_rf]
mp_rf$observed[max_diff_rf]

# Pick a single observation for which we want to gain more understanding
new_apartment <- apartments_test[max_diff_rf, ]
new_apartment

# calculate variable attribution
new_apartment_rf <- single_prediction(explainer_rf, observation = new_apartment)
new_apartment_rf # one view
breakDown:::print.broken(new_apartment_rf) # more concise view 
plot(new_apartment_rf)

new_apartment_lm <- single_prediction(explainer_lm, observation = new_apartment)
plot(new_apartment_lm, new_apartment_rf)



# lime --------------------------------------------------------------------

set.seed(304)
ind <- sample(1:nrow(iris), 5)
iris_explain <- iris[ind, 1:4]
iris_train <- iris[-ind, 1:4]
iris_lab <- iris[[5]][-ind]

library(caret)
model <- train(iris_train, iris_lab,
               method = "rf")

library(lime)
explainer <- lime(iris_train, model,
                  bin_continuous = TRUE,
                  n_bins = 4,
                  quantile_bins = TRUE)
explanation <- explain(iris_explain,
                       explainer,
                       n_labels = 1,
                       n_features = 4,
                       n_permutations = 5000,
                       feature_select = "auto")
explanation

plot_features(explanation)


# shapley -----------------------------------------------------------------

library(iml)

predictor <- Predictor$new(model, data = iris_train, y = iris_lab)
shapley <- Shapley$new(predictor, x.interest = iris_explain[1,]) 
shapley$plot()

