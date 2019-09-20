
# dalex exercise ----------------------------------------------------------

library(DALEX)
library(randomForest)
library(ggplot2)


# Load DALEX package and explore its in-built dragons and dragons_test dataset. 
# The feature to predict is life_length. 
# Create 2 explainers – one for the linear model and one for random forest .
  
str(dragons)
set.seed(519)
dragons_rf_model <- randomForest::randomForest(life_length ~ ., 
                                               data = dragons)
predicted_rf <- predict(dragons_rf_model, dragons_test)

dragons_lm_model <-  lm(life_length ~ ., data = dragons)
predicted_lm <- predict(dragons_lm_model, dragons_test)

explainer_lm <- DALEX::explain(model = dragons_lm_model, 
                               data = dragons_test[,1:7], 
                               y = dragons_test$life_length)
explainer_rf <- DALEX::explain(model = dragons_rf_model, 
                               data = dragons_test[,1:7], 
                               y = dragons_test$life_length)

# Compare models’ performance in terms of distribution of residuals. 
# Which one performs better?

sqrt(mean((predicted_rf - dragons_test$life_length)^2)) 
sqrt(mean((predicted_lm - dragons_test$life_length)^2))


mp_lm <- model_performance(explainer_lm)
mp_rf <- model_performance(explainer_rf)

plot(mp_lm, mp_rf)
plot(mp_lm, mp_rf, geom = "boxplot")

ggplot(mp_rf, 
       aes(observed, diff)) +
  geom_point() + 
  xlab("Observed") + 
  ylab("Predicted - Observed") + 
  ggtitle("Diagnostic plot") + 
  theme_mi2()


# Extract and compare models’ variable importance. 
# How are they similar? How are they different?

vi_rf <- variable_importance(explainer_rf, 
                             loss_function = loss_root_mean_square)
vi_lm <- variable_importance(explainer_lm, 
                             loss_function = loss_root_mean_square)

plot(vi_rf, vi_lm)

# Explore height in both models using Partial Dependency Plots. 
# What are the differences and why?

sv_rf  <- single_variable(explainer_rf, 
                          variable = "height", 
                          type = "pdp")
sv_lm  <- single_variable(explainer_lm, 
                          variable = "height", 
                          type = "pdp")
plot(sv_rf, sv_lm)

# Select the top outlier from the Random Forest model and understand the 
# composition of its prediction in both linear and random forest models. 
# Which algorithm performed better and why?

max_diff_rf  <- which.max(abs(mp_rf$diff))

mp_rf$predicted[max_diff_rf]
mp_rf$observed[max_diff_rf]

new_dragon <- dragons_test[max_diff_rf, ]
new_dragon

# calculate variable attribution
new_dragon_rf <- single_prediction(explainer_rf, observation = new_dragon)
new_dragon_rf # one view
breakDown:::print.broken(new_dragon_rf) # more concise view 
plot(new_dragon_rf)

new_dragon_lm <- single_prediction(explainer_lm, observation = new_dragon)
plot(new_dragon_lm, new_dragon_rf)



# lime exercise -----------------------------------------------------------

data("AdultUCI", package = "arules")

# prep data
AdultUCI <- na.omit(AdultUCI)
AdultUCI$education <- factor(AdultUCI$education, ordered = FALSE)
AdultUCI$income <- factor(AdultUCI$income, ordered = FALSE)

# split data
set.seed(304)
AdultUCI <- AdultUCI[sample(1:nrow(AdultUCI), 1000),]
test_ind <- sample(1:nrow(AdultUCI), 5)
adult_test <- AdultUCI[test_ind, 1:14]
adult_train <- AdultUCI[-test_ind, 1:14]
adult_lab <- AdultUCI[[15]][-test_ind]

# fit a model
library(caret)
system.time(
  model <- train(adult_train, adult_lab, method = "rf")
) # 73s for 1000-5 obs

# make an explainer and get explanations
explainer <- lime(adult_train, model)#, bin_continuous = FALSE)
explanation <- explain(adult_test, explainer, 
                       n_labels = 1, n_features = 5)

# easier to comprehend in a plot
plot_features(explanation)

