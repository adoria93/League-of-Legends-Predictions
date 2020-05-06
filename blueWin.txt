library(tidyverse)
library(broom)
library(corrplot)
library(car)
library(pROC)

# Data Source: https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min

# Import the Data

high_diamond_ranked_10min <- read.csv("~/Projects/05 League of Legends/00 Data/high_diamond_ranked_10min.csv")

# Creating a working dataset

league <- high_diamond_ranked_10min

summary(league)

# EDA (Exploratory Data Analysis)

average_blueLevel = round(mean(high_diamond_ranked_10min$blueAvgLevel))
average_blueGold = round(mean(high_diamond_ranked_10min$blueTotalGold))
average_blueExp = round(mean(high_diamond_ranked_10min$blueTotalExperience))

average_redLevel = round(mean(high_diamond_ranked_10min$redAvgLevel))
average_redGold = round(mean(high_diamond_ranked_10min$redTotalGold))
average_redExp = round(mean(high_diamond_ranked_10min$redTotalExperience))

# Combining all into a single table

EDA_Table <- data.frame("Team" = c("Blue", "Red"), "Average Level" = c(average_blueLevel, average_redLevel),
                        "Average Gold" = c(average_blueGold, average_redGold), "Average Experience" = c(average_blueExp, average_redExp))

# Plots

ggplot(data = EDA_Table, aes(x = EDA_Table$Team, y = EDA_Table$Average.Level)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Average Level for each Team", 
       x = "Team", y = "Average Level")

ggplot(data = EDA_Table, aes(x = EDA_Table$Team, y = EDA_Table$Average.Gold)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Average Total Gold for each Team", 
       x = "Team", y = "Average Gold")

ggplot(data = EDA_Table, aes(x = EDA_Table$Team, y = EDA_Table$Average.Experience)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Average Total Experience for each Team", 
       x = "Team", y = "Average Experience")

#------------------------------
# Cleaning the Data
#------------------------------

# Drop unnecessary variables as some are just repeated values
# (ex. blueFirstBlood/redFirstBlood, blueDeaths/redKills are just repeated values)

league <- subset(league, select = -c(gameId, blueGoldDiff, blueExperienceDiff,
                      blueGoldPerMin, blueHeralds, redFirstBlood,
                      redKills, redEliteMonsters, redDragons,
                      redGoldDiff, redExperienceDiff,
                      redGoldPerMin, redHeralds
                      ))

# The goal is to use logistic regression in order to determine
# what factors influence the blue team's odds of winning. In
# order to do this, variables that are highly correlated with
# each other need to be removed in order to avoid colinearity.

league.cor = cor(league)
corrplot(league.cor, method = "square", tl.cex = 0.5)

# Drop the correlated variables

league = subset(league, select = -c(blueTotalGold, blueAvgLevel, blueTotalExperience,
                                    redDeaths, redAssists, redTotalExperience,
                                    redTotalGold, redAvgLevel, redCSPerMin, blueCSPerMin))

# Updated correlation plot

league.cor = cor(league)
corrplot(league.cor, method = "number", tl.cex = 0.5, number.cex = 0.5)

# Checking variable classifications

str(league)

# Some of these variables are not categorized correctly. For example,
# the blueWins variable is supposed to be a factor where 0 represents
# a loss and 1 is a win, but is currently set up as an integer.

league$blueWins <- as.factor(league$blueWins)
league$blueFirstBlood <- as.factor(league$blueFirstBlood)

# Checking to see if the changes were done correctly

str(league)

#--------------------------
# Training/Test Data
#--------------------------

# I like using 80% of the data to build the model and the
# remaining data to test the model.

# Create a random variable that will be used for selection

set.seed(300)
league$random <- runif(nrow(league))

# Order by the random variable to "mix up" the games and then
# dropping that random variable from the final dataset.

league <- league[order(league$random),]
league <- subset(league, select = -c(random))

# Training data will be used to create the model, testing
# data will be used to verify how well the model works

train <- league[1:round(0.8*nrow(league), 0),]
test <- league[-(1:round(0.8*nrow(league))),]

#--------------------------------
# Creating the model
#--------------------------------

logit <- glm(blueWins ~., data = train, family = "binomial")
summary(logit)

# AIC: 8535.3
# A few variables aren't really significant, creating a new
# model that drops these nonsignificant variables

logit2 <- glm(blueWins ~ ., data = subset(train, select = -c(blueFirstBlood, blueWardsPlaced, blueWardsDestroyed, blueAssists,
                                                             blueEliteMonsters, redWardsPlaced, redWardsDestroyed)), family = "binomial")
summary(logit2)

# AIC: 8528.5
# Lower AIC means a better model. While it's not a huge difference, I believe
# that the more simple model should be used everytime

# Odds Ratios

tidy(logit2, exponentiate = TRUE, conf.level = 0.95)

# Checking the VIF of the logit2 model to see if any multicol;inearity is
# still in the model. If so, remove variables with VIF > 4 and redo the
# model.

vif(logit2)

#-----------------------
# Accuracy of the model
#-----------------------

# Using the test dataset to see how well the model did

prediction <- predict(logit2, test, type = "response")

# Need to determine a cutoff to use, can check that using a ROC Graph
# True Positive Rate -- Sensitivity
# False Positive Rate -- 1 - Specificity 

roc(train$blueWins, logit2$fitted.values, plot = TRUE, legacy.axes = TRUE,
    percent = TRUE, xlab = "False Positive Percentage", ylab = "True Positive Percentage",
    lwd = 4, print.auc = TRUE)

# Considering that we are trying to predict if the blue team will win
# a match of a video game at 10 minutes into the match, a cutoff of
# 50% should be fine.

prediction <- round(prediction)

# Create a Table (Confusion Matrix)

prediction_table <- table(Predicted = prediction, Reference = test$blueWins)
prediction_table

# Accuracy = correct / (correct + incorrect)
# Multiply by 100 to get a percentage

accuracy = ((prediction_table[1,1] + prediction_table[2,2]) /
               (prediction_table[1,1] + prediction_table[1,2] +
               prediction_table[2,1] + prediction_table[2,2])) * 100
accuracy

# 72.3% accuracy. Not bad! According to LeagueofGraphs.com,
# the average length of game is around 28 minutes for Diamond solo/duo queue.
# Considering the data gathered was collected at the 10 minute mark,
# I'd say this model did a pretty good job!

# Lots can happen after the 10 minute mark. I'm not too familiar
# with the game mechanics so it is possible that I could've
# removed a variable that would've been a good idea to keep.
# This is where experience would come in. In the idea situation,
# showing this to a person who plays league of legends would be
# great as they'd be able to share what is considered "important"
# to keep.
