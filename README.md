# NLP
# Sentiment Analysis on restaurant reviews on R language. 
# Load required packages
library(ggplot2) #Used for creating plots and visualizations.
library(caret) #Used for machine learning model training and evaluation.
library(NLP) #NLP and tm: used for natural language processing (NLP) tasks.
library(tm)
library(glmnet) #Used for fitting generalized linear models (logistic regression).
library(wordcloud) #wordcloud and wordcloud2: Used for creating word clouds.
library(wordcloud2)
library(dplyr) #dply, tidyverse and tidyr: Used for data manipulation and tidying.
library(tidyverse)
library(tidyr)
library(glmnet) #Used for fitting generalized linear models (logistic regression).

#Uploading the dataset
Data <- read_csv("C:/Users/96653/OneDrive/Desktop/Resturant/Resturant_Review.csv")
#view(Data)

# --------------Dataset description--------------------------------------------

# Checking the size of the dataset:
dim(Data)
# Number of attributes
ncol(Data)
# Types of attributes
str(Data)
# Null values
is.na(Data)
# Number of null values
sum(is.na(Data)) 

# -----------Study of the distribution of individual variables----------------- 

# Summary of all dataset
summary(Data)
# Summary of Review column
summary(Data$Review)
# Summary of Liked column
summary(Data$Liked)
# Histogram
hist(Data$Liked)
# Distribution plot
plot(density(Data$Liked)) 
# Box plot
boxplot(Data$Liked) 

# ------------Data cleaning----------------------------------------------------

# Duplicated Data
duplicated(Data)
# Number of duplicated
sum(duplicated(Data))
# Remove duplocated data
Clean_Data<- distinct(Data)
# After cleaning duplicated data
duplicated(Clean_Data)
sum(duplicated(Clean_Data))
# New dim of dataset
dim(Clean_Data)
# Missing values
any(is.na(Data)) 

# ---------------------Clean the text data-------------------------------------

# custom states (Accented letters) replaced themselves without the Accented Characters
Data$Review[151] <- "My fiance and I came in the middle of the day and we were greeted and seated right away."
Data$Review[599] <- "I really enjoyed Crema Cafe before they expanded. I even told friends they had the BEST breakfast."
Data$Review[824] <- "The crepe was delicate and thin and moist."
Data$Review[916] <- "The only thing I wasn't too crazy about was their guacamole as I don't like it pureed."


# preprocessing function 
text_preprocessing<- function(x) {
  x<- gsub('#\\S+','',x) # remove hashtags
  x<- gsub('[[:cntrl:]]','',x) # remove controls and special characters
  x<- gsub("^[[:space:]]*","",x) # remove leading whitespaces
  x<- gsub("[[:space:]]*$","",x) # remove trailing whitespaces
  x<- gsub(" +"," ", x) # remove extra whitespaces
  x<- gsub("log\\(", "", x)
  x<- gsub(",","",x) #remove comma
  x<- gsub(".$", "", x) # remove dot in the end of the sentence
  x<- gsub("[[:punct:]]", "", x) # Remove punctuation
  x<- tolower(x) # Convert to lowercase
  x<- removeWords(x, stopwords("english"))
  x<- gsub("ooo", "o", x) # Remove punctuation
}

# Create a list to store preprocessed reviews
list1 <- list()

# Loop through the dataset and apply text preprocessing
for (i in 1:nrow(Data)) {
  review_i <- text_preprocessing(Data$Review[i])
  list1[[i]] <- review_i
}

# Convert the list of preprocessed reviews to a data frame
reviews_processed <- data.frame(processed_review = unlist(list1))

#--------------Hypothesis testing----------------------------------------------
#to assess whether there is a significant difference in means between the "Score" values for the two levels of the "Liked" column.

# Converts text reviews to numeric scores by function
sentiment_score <- function(text) {
  #assume positive reviews get a score of 3, neutral get 2, and negative get 1
  if (grepl("good|loved", tolower(text))) {
    return(3)
  } else if (grepl("not good|nasty", tolower(text))) {
    return(1)
  } else {
    return(2)
  }
}
# Apply the sentiment_score function to create a new column "Score"
Data$Score <- sapply(Data$Review, sentiment_score)

# Check the new "Score" column
#head(Data$Score)

# Perform Welch  t-test on the "Score" column based on the "Liked" column
result <- t.test(Score ~ Liked, data = Data)

# Print the result
print(result)

# Interpret the results
alpha <- 0.05

if (result$p.value < alpha) {
  cat("Reject the null hypothesis. There is a significant difference in scores between Liked and Not Liked groups.\n")
} else {
  cat("Fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference in scores between Liked and Not Liked groups.\n")
}


#---------------Bag of Words and TF-IDF Matrix----------------------------------

# Create a corpus and the term-document matrix
R_Corpus <- Corpus(VectorSource(reviews_processed$processed_review))
tdm <- TermDocumentMatrix(R_Corpus, control = list(wordLengths = c(1, Inf)))

# Create a TF-IDF matrix
tfidf <- weightTfIdf(tdm)

# Convert the TF-IDF matrix to a data frame
tfidf_df <- as.data.frame(as.matrix(tfidf))
dim(tfidf_df)

#-----------------Model Building and Evaluation---------------------------------

# Split the data into training and testing sets
set.seed(123)
train_x <- tfidf_df[1:800, ]
train_y <- Data$Liked[1:800]
test_x <- tfidf_df[801:1000, ]
test_y <- Data$Liked[801:1000]

train_data <- data.frame(Liked = train_y, train_x)
test_data <- data.frame(Liked = test_y, test_x)

# Build a logistic regression model
model <- glm(Liked ~ ., data = train_data, family=binomial(link="logit"), maxit = 1000)

# Assuming 'Liked' in train_data has levels 0 and 1
train_data$Liked <- factor(train_data$Liked, levels = c(0, 1))

# Assuming 'Liked' in test_data has the same levels as in train_data
test_data$Liked <- factor(test_data$Liked, levels = levels(train_data$Liked))

# Make predictions on the test set
predictions <- predict(model, newdata = test_data, type = "response")

# Convert predicted probabilities to binary predictions (0 or 1)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Assuming 'Liked' has levels 0 and 1
predicted_classes <- factor(predicted_classes, levels = levels(test_data$Liked))

# Subset predictions to match the length of test_data$Liked
predictions <- predicted_classes[1:200]

# Evaluate the model
confusionMatrix(predictions, test_data$Liked)

#------------------ visiualization---------------------------------------------

# Inspect frequent words
freq_terms <- findFreqTerms(tdm, lowfreq = 50)
#View(freq_terms)


# inspect frequent words
term_freq<- rowSums(as.matrix(tdm))
term_freq<- subset(term_freq, term_freq>=20)
df<- data.frame(term = names(term_freq), freq = term_freq)
#View(df)

# plot word frequency
df_plot<- df %>%
  top_n(25)
# Plot word frequency
ggplot(df_plot, aes(x = reorder(term, +freq), y = freq, fill = freq)) + geom_bar(stat = "identity")+ scale_colour_gradientn(colors = terrain.colors(10))+ xlab("Terms")+ ylab("Count")+coord_flip()

# create word cloud
m<- as.matrix(tdm)
# calculate the frequency of words as sort it by frequency
word_freq<- sort(rowSums(m), decreasing = T)
wordcloud2(df, color = "random-dark", backgroundColor = "white")

#-------------------------End---------------------------------------------------

