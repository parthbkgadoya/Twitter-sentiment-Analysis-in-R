tweets = read.csv('tweets.csv', stringsAsFactors = FALSE)
head(tweets)
str(tweets)

# the data(tweets) is not processed
tweets$negative = as.factor(tweets$Avg <= -1)
table(tweets$negative)
#install.packages('tm')
library(tm)
#install.packages('SnowballC')
library(SnowballC)
### tweets was the dataFrame
corpus = Corpus(VectorSource(tweets$Tweet))
corpus
corpus[[1]]$content

## After making corpus, we are ready to do pre-processing on our data.
#   first, we all convert all data into lowercase.
corpus = tm_map(corpus, FUN = tolower)
corpus[[1]]$content

#  Now, we will remove all puntuations from the corpus
corpus = tm_map(corpus, FUN = removePunctuation)
corpus[[1]]$content

#  Now, we will remove stopwords from the tweets in corpus
stopwords('en')[1:10]
corpus = tm_map(corpus, removeWords, c('apple', stopwords('en') ) )
corpus[[1]]$content

# Now, we will use stemming on our tweets
corpus = tm_map(corpus, FUN = stemDocument)
corpus[[1]]$content

#### Applying bag of words on our corpus
frequencies = DocumentTermMatrix(corpus)
frequencies
inspect(frequencies[1000:1005,505:515])

findFreqTerms(frequencies, lowfreq = 20)
## only 56 terms apper 20 times in our matrix
# this suggested that we have a lot of useless terms in our document
# this is not good for 2 reasons
# 1. the ratio of independent terms to the number of observations will decide the 
#    accuracy of model
# 2. More terms means more computaion power needed.

# Therefore, we will remove some these unwanted terms.
# The data(matix) "frequencies" is a sparse one. i.e. it has many zeroes.

sparse = removeSparseTerms(frequencies, 0.995 )
sparse

tweetSparse = as.data.frame(as.matrix(sparse))
head(tweetSparse)
# R struggles with variable names starting from numbers.
# We will use make.names functio to deal with this.
colnames(tweetSparse) = make.names(colnames(tweetSparse))
tweetSparse$negative =  tweets$negative

# We now have our dataframe to work with.
# We will first split the data into training and testing set.
library(caTools)
set.seed(123)
split = sample.split(tweetSparse$negative,SplitRatio = 0.7)

trainSparse = subset(tweetSparse, split == TRUE)
testSparse = subset(tweetSparse, split == FALSE)


## We will now make the predictive model
library(rpart)
library(rpart.plot)
tweetCART = rpart(negative ~ ., data = trainSparse, method = 'class')
prp(tweetCART)

predictCART = predict(tweetCART, newdata = testSparse, type = 'class')
table(testSparse$negative, predictCART)
# accuracy comes out to be 0.88
## now, baseline model will give
table(testSparse$negative)
# accuracy is 0.845

### How will the random forest model work
# let us check it out
library(randomForest)
tweetRF = randomForest(negative ~ ., data = trainSparse)
predictRF = predict(tweetRF, newdata = testSparse)
table(testSparse$negative, predictRF)
## Accuracy comes out to be 0.89

## This accuracy is better than CART by 1%, but due interpretablity I will choose
## CART over random forest. I can also increase accuracy of CART by using CP 
## value through cross validation.

#### We successfully predicted sentiments of tweet even with relatively small
#### number of tweets.


## lets also apply logistic regression and see its performance
tweetLOG = glm(negative ~ ., data = trainSparse, family = 'binomial')
predictLOG = predict(tweetLOG, newdata = testSparse, type = 'response')

table(testSparse$negative, predictLOG > 0.5)
## Accuracy of logistic regression model comes out to be 0.81
## This is even worse than the baseline model.
