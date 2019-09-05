# dalex -------------------------------------------------------------------

library(DALEX)
library(randomForest)
library(ggplot2)

# load data
data("apartments")
data("apartments_test")

# train random forest and linear model
head(apartments)
set.seed(519)

## train RF model
apartments_rf_model <- 
  randomForest::randomForest(m2.price ~ ., 
                             data = apartments)

## train LM model
apartments_lm_model <-  lm(m2.price ~ ., 
                           data = apartments)

# predict on test set using RF
predicted_rf <- predict(apartments_rf_model, 
                        apartments_test)

# predict on test set using LM
predicted_lm <- predict(apartments_lm_model, 
                        apartments_test)


# root mean square 
sqrt(mean((predicted_rf - apartments_test$m2.price)^2)) 
sqrt(mean((predicted_lm - apartments_test$m2.price)^2))
# --> very similar --> let's explore the models in more detail

# Many models available in R, but not a standard API
# good amount of standardisation via caret and mlr
# DALEX works on both plus many "individual" models
# First create an explainer obj for DALEX to standardise
# That's the input obj for most dalex functions!

# create dalex explainer object
explainer_lm <- DALEX::explain(model = apartments_lm_model, 
                               data = apartments_test[,2:6], 
                               y = apartments_test$m2.price)
explainer_rf <- DALEX::explain(model = apartments_rf_model, 
                               data = apartments_test[,2:6], 
                               y = apartments_test$m2.price)

# show object
explainer_rf



# dalex: How good is the model? > Model performance audit -----------------

mp_lm <- model_performance(explainer_lm)
mp_rf <- model_performance(explainer_rf)


# Object contains residuals
# - explore yourself 
# - or use canned plots (many dalex objects have a plot method)
mp_rf

# plot absolute residuals

# via inverse cdf 
plot(mp_lm, mp_rf)
# RF has bigger spread of residuals: more small ones but also more bigger ones

# via boxplot
plot(mp_lm, mp_rf, geom = "boxplot")



# dalex: How good is the model? > Goodness of fit -------------------------

# mp object contains predicted, observed, diff (=residuals)
str(mp_rf)

ggplot(mp_rf, 
       aes(observed, diff)) +
  geom_point() + 
  xlab("Observed") + 
  ylab("Predicted - Observed") + 
  ggtitle("Diagnostic plot") + 
  theme_mi2()
# --> RF underestimates for high value flats

# Same for LM shows two groups of residuals! --> reason for bumps in inverse cdf!
# due to effect of construction.year being non-linear?

# ~~~~~~~~ exercise ~~~~~~~~



# dalex: How does the model work? > All variables -------------------------

# idea: 
# - calculate a loss for the model (as is)
# - perturb a variable, refit the model, re-calculate the loss
# - variable importance: change in loss
# - baseline is loss for a model with perturbed response (see when printing)

vi_rf <- ingredients::feature_importance(
  explainer_rf,
  loss_function = loss_root_mean_square)

vi_lm <- ingredients::feature_importance(
  explainer_lm,
  loss_function = loss_root_mean_square)

vi_rf

plot(vi_rf, vi_lm)
# NOTE difference in importance of construction.year


# ~~~~~~~~ exercise ~~~~~~~~


# dalex: How does the model work? > Continuous variable -------------------

# partial dependency plots:
# - show marginal effect of a feature on the predicted response
# - can show form of relationship: linear? non-linear? monotonous?

sv_rf  <- single_variable(explainer_rf, 
                          variable = "construction.year", 
                          type = "pdp")
sv_lm  <- single_variable(explainer_lm, 
                          variable = "construction.year", 
                          type = "pdp")
plot(sv_rf, sv_lm)

# effect of number of rooms ("no.rooms") is also more sigmoid than linear


# dalex: How does the model work? > Categorical variable ------------------

# - visualization of post-hoc comparisons between factor levels via LR test
# - idea: which factor levels can be grouped?

# Warsaw knowledge:
# - Srodmiescie is the city centre
# - well connected to the centre: Ochota, Mokotow, Zoliborz
# - rest: further out from the city centre

# extra:
# - Stars present how significant are differences between the two clusters. 
# - The numbers present the value of prediction: Intercept + factor_level

svd_rf  <- single_variable(explainer_rf, 
                           variable = "district", 
                           type = "factor")
svd_lm  <- single_variable(explainer_lm, 
                           variable = "district", 
                           type = "factor")
plot(svd_rf, svd_lm)



# ~~~~~~~~ exercise ~~~~~~~~

# lime --------------------------------------------------------------------

library(lime)
library(caret)

# explore iris dataset
pairs(iris[1:4], pch = 21, 
      bg = c("#1b9e77", "#d95f02", "#7570b3")[unclass(iris$Species)])

## create train/test sets 
set.seed(304)
ind <- sample(1:nrow(iris), 5)
iris_explain <- iris[ind, 1:4]
iris_train <- iris[-ind, 1:4]
iris_lab <- iris[[5]][-ind]

# train random forest
model <- train(iris_train, iris_lab,
               method = "rf")

# create a LIME explainer 
explainer <- lime(iris_train, model,
                  bin_continuous = TRUE,
                  n_bins = 4,
                  quantile_bins = TRUE)

# create explanations
explanation <- explain(iris_explain,
                       explainer,
                       n_labels = 1,
                       n_features = 4,
                       n_permutations = 5000,
                       feature_select = "auto")
# explore explanations
explanation
head(eexplanation)
View(explanation)

# plot explanations
plot_features(explanation)


# shapley -----------------------------------------------------------------
library(caret)
library(iml)

## re-run LIME code below if you start with the clean session
### 
set.seed(304)
ind <- sample(1:nrow(iris), 5)
iris_explain <- iris[ind, 1:4]
iris_train <- iris[-ind, 1:4]
iris_lab <- iris[[5]][-ind]

model <- train(iris_train, iris_lab, method = "rf")
###


# create a predictor
predictor <- Predictor$new(model, 
                           data = iris_train, 
                           y = iris_lab)
# calculate Shapley values
shapley <- Shapley$new(predictor, 
                       x.interest = iris_explain[1,]) 

# plot Shapley values
shapley$plot()