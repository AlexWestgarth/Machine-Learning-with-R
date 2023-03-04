#
# Import Libraries
library(tidyverse)
library(ROSE)
library(dplyr)
library(caret)
library(quanteda)
library(caTools)
library(tm)
library(wordcloud)
library(SnowballC)
library(gridExtra)
library(tidytext)
library(RColorBrewer)
library(ggplot2)
library(gmodels)
library(wordcloud2)
library(e1071)
library(pROC)
library(ROCR)
library(rvest)
library(stringr)
library(dplyr)
library(colorRamps)
library(tau) 
library(data.table)
library(smotefamily)
library(textstem)
library(naivebayes)
library(randomForest)
# Set a fixed seed for this script
set.seed(215)

# Load data
data1 <- read.csv(file.choose(), header=T, sep = ";")
summary(data1)
as_tibble(data1)

# Check for issues in the target variable
ggplot(data1,aes(x = Label,fill=Label)) + 
  geom_bar()
unique(data1$Label)

# Label has 3 options, "FAKE", "REAL" and a blank ""
# The third blank option is of no use to us, as we cannot use it to test or train a model.
# Since we cannot derive the true value
# Copy data into new slot to avoid destroying raw data
data2 <- data1[!(data1$Label ==""),]
summary(data2)
unique(data2$Label)


ggplot(data2,aes(x = Label,fill=Label)) + 
  geom_bar()

# Now to factor our target
data2$Label <- as.factor(data2$Label)
# Shows the proportions of the values of the type attribute:
prop.table(table(data2$Label))

# Set new column, text length
data2$TextLength <-nchar(data2$Text)
summary(data2$TextLength)


# Allocate fake and real data
indexes <-which(data2$Label=="REAL")
real <- data2[indexes,]
fake <- data2[-indexes,]
fake$TextLength <- nchar(fake$Text)
real$TextLength <- nchar(real$Text)
summary(real$TextLength)
summary(fake$TextLength)


ggplot(data2, aes(x = TextLength, fill=Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x= "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")


# Check the changes
ggplot(data2, aes(x = Label, fill=Label)) + 
  geom_bar()
count(data2, Label)

# Text_tag investigation
# Any instance of a text_tag with multiple tags is separated via a comma.
# By unlisting and splitting these values, we can produce a list that contains each tag individually.
fake_taglist <- unlist(strsplit(fake$Text_Tag, ","))
real_taglist <- unlist(strsplit(real$Text_Tag, ","))

# Acquire the top ten of each list
topten_fake_taglist <- sort(table(fake_taglist), decreasing=TRUE)[1:10]
topten_real_taglist <- sort(table(real_taglist), decreasing=TRUE)[1:10]

# Convert from table to dataframe for graphing
df_topten_fake_taglist <- as.data.frame(topten_fake_taglist)
df_topten_real_taglist <- as.data.frame(topten_real_taglist)

# Graph
ggplot(df_topten_fake_taglist, aes(x = fake_taglist, y = Freq)) + 
  geom_bar(stat = "identity")
ggplot(df_topten_real_taglist, aes(x = real_taglist, y = Freq)) + 
  geom_bar(stat = "identity")

# A quick survey indicates that the top ten tags have no easily identifiable difference between real and fake.
# Given this fact, it may not be worthwhile to consider the text_tag variable for modelling.

# Author investigation

# Prior to any investigation, some assumptions apply:
# This investigation will only shed light on any data that contains authors existing within the current dataset
# If we were to try and apply the author trained model on data that contained new authors, the model wouldn't
# be capable of deriving anything without applying some level of inherent unwanted bias on the author's name.
# Despite this, an investigation will be placed to see if authors appear across both lists.

total_authors <- unique(data2$Author) #1000 Authors (Nice clean number)
total_fake_author <- unique(fake$Author) # 983 Authors
total_real_author <- unique(real$Author) # 996 Authors

# It appears that across both fake and real data, the authors are relatively shared.
# That, combined with the previously mentioned concern leads to the discarding of this data when modelling.

# Date investigation

# It's hard to imagine how the date of an article would influence the authenticity.
# As with authors, this investigation will focus on any exceptional outliers pertaining to unique dates.
# The default format is yyyy-mm-dd which is how the data appears to be organised.

datefake <- sort(as.Date(fake$Date, format="%Y/%m/%d"))
datereal <- sort(as.Date(real$Date, format="%Y/%m/%d"))
setdiff(unique(datefake), unique(datereal)) 
setdiff(unique(datereal), unique(datefake)) 


corpus <- VCorpus(VectorSource(data2$Text))

# Convert text to lower case
corpus <- tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus <- tm_map(corpus, removeNumbers)

# Remove Punctuations
corpus <- tm_map(corpus, removePunctuation)

# Remove Stopwords
corpus <- tm_map(corpus, removeWords, stopwords('english'))

# Remove Whitespace
corpus <- tm_map(corpus, stripWhitespace)

# Lemmatization
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))

# Convert to DTM
dtm <- DocumentTermMatrix(corpus)
inspect(dtm)

# Clean DTM of spare terms
dtm_clean <- removeSparseTerms(dtm, sparse = 0.99)
inspect(dtm_clean)

# Tidy DTM
df_tidy <- tidy(dtm_clean)
df_word<- df_tidy %>% 
  select(-document) %>%
  group_by(term) %>%
  summarize(freq = sum(count)) %>%
  arrange(desc(freq))

# Wordcloud
wordcloud2(data=df_word, size=1.6, color='random-dark')

# Convert to matrix
dtm_mat <- as.matrix(dtm_clean)
dim(dtm_mat)

dtm_mat <- cbind(dtm_mat, Label = data2$Label)
dtm_mat[1:10, c(1, 2, 3, ncol(dtm_mat))]
summary(dtm_mat[, 'Label'])

# Compare label from Matrix to Original Data
as.data.frame(dtm_mat) %>% count(Label)
data2 %>% count(Label)


dtm_df <- as.data.frame(dtm_mat)

dtm_df$Label <- ifelse(dtm_df$Label == 2, 1, 0)
dtm_df$Label <- as.factor(dtm_df$Label)
table(dtm_df$Label)
# Create 75:25 split
index <- sample(nrow(dtm_df), nrow(dtm_df)*0.75, replace = FALSE)

train <- dtm_df[index,]
test <- dtm_df[-index,]

# make column names to follow R's variable naming convention
names(train) <- make.names(names(train))
names(test) <- make.names(names(test))

table(train$Label)
table(test$Label)

# Perform sampling here
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Label, 
                         yname = "Label")
table(down_train$Label)

# upsampling
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Label,
                     yname = "Label")
table(up_train$Label)

# Smote is not advised for non-numerical data and so will not be performed on
# This dataset.
# ROSE only accepts continuous or categorical variables, and is also unsuitable
# for this dataset.


# Naive Bayes Model
train_nb <- naive_bayes(Label ~ ., data = train)
train_up_nb <- naive_bayes(Label ~ ., data = up_train)
train_down_nb <- naive_bayes(Label ~ ., data = down_train)

# Model Summary
summary(train_nb)
summary(train_up_nb)
summary(train_down_nb)


# Logistic Regression Model
train_lr <- glm(formula = Label ~.,
              data = train,
              family = 'binomial')
train_up_lr <- glm(formula = Label ~.,
                data = up_train,
                family = 'binomial')
train_down_lr <- glm(formula = Label ~.,
                data = down_train,
                family = 'binomial')
# Random Forest Model
train_k <- round(sqrt(ncol(train)-1))
train_up_k <- round(sqrt(ncol(up_train)-1))
train_down_k <- round(sqrt(ncol(down_train)-1))
train_up_rf <- randomForest(formula = Label ~ ., 
                       data = up_train,
                       ntree = 100,
                       mtry = train_up_k,
                       method = 'class')
train_down_rf <- randomForest(formula = Label ~ ., 
                         data = down_train,
                         ntree = 100,
                         mtry = train_down_k,
                         method = 'class')
train_rf <- randomForest(formula = Label ~ ., 
                         data = train,
                         ntree = 100,
                         mtry = train_k,
                         method = 'class')

train_rf
train_up_rf
train_down_rf

# Predicted values
train$pred_nb <- predict(train_nb, type = 'class')
up_train$pred_nb <- predict(train_up_nb, type = 'class')
down_train$pred_nb <- predict(train_down_nb, type = 'class')


train$pred_lr <- predict(train_lr, type = 'response')
up_train$pred_lr <- predict(train_up_lr, type = 'response')
down_train$pred_lr <- predict(train_down_lr, type = 'response')


train$pred_rf <- predict(train_rf, type = 'response')
up_train$pred_rf <- predict(train_up_rf, type = 'response')
down_train$pred_rf <- predict(train_down_rf, type = 'response')


# Predicted Values for test set
test$pred_nb <- predict(train_nb, newdata = test)
test$pred_nb_up <- predict(train_up_nb, newdata = test)
test$pred_nb_down <- predict(train_down_nb, newdata = test)

test$pred_lr <- predict(train_lr, newdata = test, type = 'response')
test$pred_lr_up <- predict(train_up_lr, newdata = test, type = 'response')
test$pred_lr_down <- predict(train_down_lr, newdata = test, type = 'response')

test$pred_rf <- predict(train_rf, newdata = test, type = 'response')
test$pred_rf_up <- predict(train_up_rf, newdata = test, type = 'response')
test$pred_rf_down <- predict(train_down_rf, newdata = test, type = 'response')

# Plot ROC Curve for sampling
prediction(as.numeric(train$pred_nb), as.numeric(train$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(col = 'red', lwd = 2)

prediction(as.numeric(up_train$pred_nb), as.numeric(up_train$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(add = TRUE, col = 'blue', lwd = 2)

prediction(as.numeric(down_train$pred_nb), as.numeric(down_train$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(add = TRUE, col = 'green', lwd = 2)

legend(0.7, 0.2, legend=c("Unbalanced", "Upsampled", "Downsampled"),
       col=c("red", "blue", 'green'), lty = 1, cex = 1.2, box.lty = 0)

# AUC for sampling
roc_nb =prediction(as.numeric(train$pred_nb), as.numeric(train$Label))
roc_nb_up =prediction(as.numeric(up_train$pred_nb), as.numeric(up_train$Label))
roc_nb_down =prediction(as.numeric(down_train$pred_nb), as.numeric(down_train$Label))
auc_nb <- performance(roc_nb, measure = "auc")
auc_nb <- auc_nb@y.values[[1]]
auc_nb_up <- performance(roc_nb_up, measure = "auc")
auc_nb_up <- auc_nb_up@y.values[[1]]
auc_nb_down <- performance(roc_nb_down, measure = "auc")
auc_nb_down <- auc_nb_down@y.values[[1]]
auc_nb
auc_nb_up
auc_nb_down

# Upsampling produces best results

# Plot ROC Curve for train set
prediction(as.numeric(train$pred_nb), as.numeric(train$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(col = 'red', lwd = 2)

prediction(as.numeric(train$pred_lr), as.numeric(train$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(add = TRUE, col = 'blue', lwd = 2)

prediction(as.numeric(train$pred_rf), as.numeric(train$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(add = TRUE, col = 'green', lwd = 2)

legend(0.8, 0.2, legend=c("NB", "Logistic", "RF"),
       col=c("red", "blue", 'green'), lty = 1, cex = 1.2, box.lty = 0)

# Plot ROC Curve for test set
prediction(as.numeric(test$pred_nb), as.numeric(test$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(col = 'red', lwd = 2)

prediction(as.numeric(test$pred_lr), as.numeric(test$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(add = TRUE, col = 'blue', lwd = 2)

prediction(as.numeric(test$pred_rf), as.numeric(test$Label)) %>%
  performance('tpr', 'fpr') %>%
  plot(add = TRUE, col = 'green', lwd = 2)

legend(0.8, 0.2, legend=c("NB", "Logistic", "RF"),
       col=c("red", "blue", 'green'), lty = 1, cex = 1.2, box.lty = 0)

roc_nb =prediction(as.numeric(test$pred_nb), as.numeric(test$Label))
roc_lr =prediction(as.numeric(test$pred_lr), as.numeric(test$Label))
roc_rf =prediction(as.numeric(test$pred_rf), as.numeric(test$Label))
auc_nb <- performance(roc_nb, measure = "auc")
auc_nb <- auc_nb@y.values[[1]]
auc_lr <- performance(roc_lr, measure = "auc")
auc_lr <- auc_lr@y.values[[1]]
auc_rf <- performance(roc_rf, measure = "auc")
auc_rf <- auc_rf@y.values[[1]]
auc_nb
auc_lr
auc_rf

roc(test$Label, test$pred_lr) %>% coords()

test$pred_lr <- ifelse(test$pred_lr > 0.5, 1, 0)
test$pred_lr <- as.factor(test$pred_lr)


conf_nb <- caret::confusionMatrix(test$Label, test$pred_nb)
conf_nb_up <- caret::confusionMatrix(test$Label, test$pred_nb_up)
conf_nb_down <- caret::confusionMatrix(test$Label, test$pred_nb_down)
conf_nb
conf_nb_up
conf_nb_down
conf_lr <- caret::confusionMatrix(test$Label, test$pred_lr)
conf_rf <- caret::confusionMatrix(test$Label, test$pred_rf)

bind_rows(as.data.frame(conf_nb$table), as.data.frame(conf_nb_up$table), as.data.frame(conf_nb_down$table)) %>% 
  mutate(Model = rep(c('Naive Bayes Raw', 'Upsampled', 'Downsampled'), each = 4)) %>%
  ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  labs(x = 'Actual', y = 'Predicted') +
  scale_fill_gradient(low = "#CCE5FF", high = "#000099") +
  scale_x_discrete(limits = c('1', '0'), labels = c('1' = 'Not Fake', '0' = 'Fake')) +
  scale_y_discrete(labels = c('1' = 'Not Fake', '0' = 'Fake')) +
  facet_grid(. ~ Model) +
  geom_text(aes(label = Freq), fontface = 'bold') +
  theme(panel.background = element_blank(),
        legend.position = 'none',
        axis.line = element_line(colour = "black"),
        axis.title = element_text(size = 14, face = 'bold'),
        axis.text = element_text(size = 11, face = 'bold'),
        axis.text.y = element_text(angle = 90, hjust = 0.5),
        strip.background = element_blank(),
        strip.text = element_text(size = 12, face = 'bold'))

bind_rows(as.data.frame(conf_nb$table), as.data.frame(conf_lr$table), as.data.frame(conf_rf$table)) %>% 
  mutate(Model = rep(c('Naive Bayes', 'Logistic Regression', 'Random Forest'), each = 4)) %>%
  ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  labs(x = 'Actual', y = 'Predicted') +
  scale_fill_gradient(low = "#CCE5FF", high = "#000099") +
  scale_x_discrete(limits = c('1', '0'), labels = c('1' = 'Not Fake', '0' = 'Fake')) +
  scale_y_discrete(labels = c('1' = 'Not Fake', '0' = 'Fake')) +
  facet_grid(. ~ Model) +
  geom_text(aes(label = Freq), fontface = 'bold') +
  theme(panel.background = element_blank(),
        legend.position = 'none',
        axis.line = element_line(colour = "black"),
        axis.title = element_text(size = 14, face = 'bold'),
        axis.text = element_text(size = 11, face = 'bold'),
        axis.text.y = element_text(angle = 90, hjust = 0.5),
        strip.background = element_blank(),
        strip.text = element_text(size = 12, face = 'bold'))
# Sampling Scoring
acc <- c(nb = conf_nb[['overall']]['Accuracy'], 
         lr = conf_nb_up[['overall']]['Accuracy'],
         rf = conf_nb_down[['overall']]['Accuracy'])

precision <- c(nb = conf_nb[['byClass']]['Pos Pred Value'], 
               lr = conf_nb_up[['byClass']]['Pos Pred Value'], 
               rf = conf_nb_down[['byClass']]['Pos Pred Value'])

recall <- c(nb = conf_nb[['byClass']]['Sensitivity'], 
            lr = conf_nb_up[['byClass']]['Sensitivity'],
            rf = conf_nb_down[['byClass']]['Sensitivity'])

data.frame(Model = c('Unbalanced', 'Upsampled', 'Downsampled'),
           Accuracy = acc,
           F1_Score = (2 * precision * recall) / (precision + recall),
           row.names = NULL)

# Model Scoring
acc <- c(nb = conf_nb[['overall']]['Accuracy'], 
         lr = conf_lr[['overall']]['Accuracy'],
         rf = conf_rf[['overall']]['Accuracy'])

precision <- c(nb = conf_nb[['byClass']]['Pos Pred Value'], 
               lr = conf_lr[['byClass']]['Pos Pred Value'], 
               rf = conf_rf[['byClass']]['Pos Pred Value'])

recall <- c(nb = conf_nb[['byClass']]['Sensitivity'], 
            lr = conf_lr[['byClass']]['Sensitivity'],
            rf = conf_rf[['byClass']]['Sensitivity'])

data.frame(Model = c('Naive Bayes', 'Logistic Regression', 'Random Forest'),
           Accuracy = acc,
           F1_Score = (2 * precision * recall) / (precision + recall),
           row.names = NULL)

# Upsample with Logisitic regression.