dataset <- read.csv(file = 'C:/Users/Genius/Downloads/Assignment 2/Churn Dataset.csv')
head(dataset )
df=as.data.frame(do.call(cbind, dataset)) 
head(df)




install.packages("corrplot")
library(corrplot)
install.packages("reshape2")
library(reshape2)
install.packages("ggplot2")
library(ggplot2)


corrplot(cor(df[ ,sapply(df, is.numeric)] ,method = "pearson"),diag = FALSE,
         method = "ellipse",
         tl.cex = 0.7, tl.col = "black", cl.ratio = 0.2
)


#-------------------------------------------------------------------------------------------------
#load reshape2 package to use melt() function
library(reshape2)

#melt data into long format
melt_ChurnDSet<- melt(cor(dataset[, sapply(dataset, is.numeric)],
                          method = "pearson"))
#create heatmap using rescaled values
ggplot(melt_ChurnDSet, aes(Var1, Var2)) +
  geom_tile(aes(fill = value), colour = "white") +
  scale_fill_gradient(low = "white", high = "red")









#----------------------------------------------------------------------------------------------------
#convert categorical values to numerical values
df$gender<- as.numeric(as.factor(df$gender))
df$Partner<- as.numeric(as.factor(df$Partner))
df$Dependents<- as.numeric(as.factor(df$Dependents))
df$PhoneService<- as.numeric(as.factor(df$PhoneService))
df$MultipleLines<- as.numeric(as.factor(df$MultipleLines))
df$InternetService<- as.numeric(as.factor(df$InternetService))                              
df$OnlineSecurity<- as.numeric(as.factor(df$OnlineSecurity))
df$OnlineBackup<- as.numeric(as.factor(df$OnlineBackup))
df$DeviceProtection<- as.numeric(as.factor(df$DeviceProtection)) 
df$TechSupport<- as.numeric(as.factor(df$TechSupport))                                 
df$StreamingTV<- as.numeric(as.factor(df$StreamingTV))
df$StreamingMovies<- as.numeric(as.factor(df$StreamingMovies))
df$PaperlessBilling<- as.numeric(as.factor(df$PaperlessBilling))
df$PaymentMethod<- as.numeric(as.factor(df$PaymentMethod))
df$Contract<- as.numeric(as.factor(df$Contract))
df$Churn<- as.numeric(as.factor(df$Churn))
head(df)     
#------------------------------------------------------------------------------------------
#removing customerID and totalcharges columns
install.packages("dplyr")
library(dplyr)
df<-df %>% select(-customerID)
df<-df %>% select(-TotalCharges)
                                
sum(is.na(df))
colSums(is.na(df))

install.packages("caTools")
library(caTools)
install.packages("rpart")
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("caret")
library(caret)
install.packages("dplyr")
library(dplyr)
install.packages("lattice")
library(lattice)
install.packages("party")
library(party)
str(df)                             
df$gender <- factor(df$gender)
df$tenure<- as.numeric(df$SeniorCitizen)
df$SeniorCitizen<- as.numeric(df$tenure)
df$MonthlyCharges<- as.numeric(df$MonthlyCharges)
df$Partner <- factor(df$Partner)
df$PhoneService <- factor(df$PhoneService)
df$MultipleLines <- factor(df$MultipleLines)
df$InternetService <- factor(df$InternetService)
df$OnlineSecurity <- factor(df$OnlineSecurity)
df$OnlineBackup <- factor(df$OnlineBackup)
df$DeviceProtection <- factor(df$DeviceProtection)
df$TechSupport <- factor(df$TechSupport)
df$StreamingTV <- factor(df$StreamingTV)
df$StreamingMovies <- factor(df$StreamingMovies)
df$PaperlessBilling  <- factor(df$PaperlessBilling )
df$Churn <- factor(df$Churn)


#-----------------------------------------------------------------------------
set.seed(42)
sample_split <- sample.split(Y = df$Churn, SplitRatio = 0.80)
train_set <- subset(x = df, sample_split == TRUE)
test_set <- subset(x = df, sample_split == FALSE)

model <- rpart(Churn ~ ., data = train_set, method = "class") #specify method as class since we are dealing with classification
model
#plot the model
rpart.plot(model)


#Make predictions
preds <- predict(model, newdata = test_set, type = "class") #use the predict() function and pass in the testing subset
preds

#Print the confusion Matrix
confusionMatrix(test_set$Churn, preds) #check accuracy


install.packages("pROC")
library(pROC)
roc(test_set$Churn, as.numeric(as.character(preds)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#000000", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("model_name"), col=c("#000000"), lwd=4)

#---------------------------------------------------------------------------------------------------------------------
#Try different ways to improve the decision tree algorithm (e.g., use different splitting strategies, prune tree after splitting)
#-----Splitting by information gain
decisionTreeInformation <- rpart(Churn ~ ., data = train_set , method = "class", parms = list(split = "information")) #specify method as class since we are dealing with classification
decisionTreeInformation
#Make predictions
preds <- predict(decisionTreeInformation, newdata = test_set, type = "class") #use the predict() function and pass in the testing subset
preds

#Print the confusion Matrix
confusionMatrix(test_set$Churn, preds) #check accuracy

#-----Splitting by Gini index
decisionTreeGini <- train(Churn ~ . , data = train_set,
                          method = "rpart",
                          parms = list(split = "gini"), 
                        
                          tuneLength = 100)
#Make predictions
preds <- predict(decisionTreeGini, newdata = test_set, type = "class") #use the predict() function and pass in the testing subset
preds

#Print the confusion Matrix
confusionMatrix(test_set$Churn, preds) #check accuracy


#--------- Purning 
decitionTreePrune <- rpart(Churn ~ ., data = train_set, method = "class", 
                           control = rpart.control(cp = 0.0082, maxdepth = 3,minsplit = 2))
#Make predictions
preds <- predict(decitionTreePrune, newdata = test_set, type = "class") #use the predict() function and pass in the testing subset
preds

#Print the confusion Matrix
confusionMatrix(test_set$Churn, preds) #check accuracy


#-------------------------------------------------------------------------------------------------------------------------------------
#Classify the data using the XGBoost model with nrounds = 70 and max depth = 3. Evaluate the performance. 
install.packages("xgboost")
library(xgboost)
install.packages("caret")
library(caret)
install.packages("e1071")
library(e1071)
str(train_set)
X_train = train_set[,-19]
X_train<- matrix(unlist(X_train),ncol=18,nrow=5634)                # independent variables for train
y_train = train_set[,19]                                # dependent variables for train

X_test = test_set[,-19]
X_test <- matrix(unlist(X_test),ncol=18,nrow=1409)                     # independent variables for test
y_test = test_set[,19]                                   # dependent variables for test

# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

#Step 4 - Create a xgboost model
# train a model using our training data
model <- xgboost(data = xgboost_train,                    # the data   
                 max.depth=3,                           # max depth 
                 nrounds=50)                              # max number of boosting iterations

summary(model)
pred_test=predict(model,xgboost_test)
pred_test

pred_test[(pred_test>3)]=3
pred_y=as.factor((levels(y_test))[round(pred_test)])
print(pred_y)

conf_mat=confusionMatrix(y_test,pred_y)
print(conf_mat)


roc(test_set$Churn, as.numeric(as.character(pred_y)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#000000", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("model_name"), col=c("#000000"), lwd=4)

#-------------------------------------------------------------------------------------------------------------
install.packages('devtools')

devtools::install_github("rstudio/keras")
devtools::install_github("rstudio/reticulate")

install.packages("tensorflow")

install.packages('reticulate')
install.packages('keras')
library(reticulate)
library(keras)
library(tensorflow)

#reticulate::use_python("C:/Users/Genius/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0")
reticulate::use_python("C:/Users/Genius/Anaconda3/")
install_tensorflow()


#defining a keras sequential model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 19, input_shape = 784) %>%
  layer_dropout(rate=0.5)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 50) %>%
  layer_activation(activation = 'softmax')
  layer_dense(units = 2) %>%
  layer_activation(activation = 'softmax')


#compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


#fitting the model on the training dataset
model %>% fit(X_train , y_train , epochs = 50, batch_size = 128)

#Evaluating model on the cross validation dataset
loss_and_metrics <- model %>% evaluate(X_test , y_test, batch_size = 128)


preds_DNN <- predict(model , newdata = test_set, type = "class") #use the predict() function and pass in the testing subset
preds_DNN

#Print the confusion Matrix
confusionMatrix(test_set$Churn, preds_DNN) #check accuracy

#ROC graph For DNN
roc(test_set$Churn, as.numeric(as.character(preds_DNN)), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#000000", lwd=4, print.auc=TRUE)
legend("bottomright", legend=c("model_name"), col=c("#000000"), lwd=4)

#------------------------------------------------------------------------------------------------------
#Part (B) in the Assigment 
#first reading the transactions 

install.packages("arules")
library(arules)
install.packages("arulesViz")
library(arulesViz)
install.packages("readr")
library(readr)
install.packages("RColorBrewer")
library(RColorBrewer)

d <- read.transactions('C:/Users/Genius/Downloads/Assignment 2/transactions.csv', format = 'basket', sep = ',')
head(d )
typeof(d)
summary(d)
plot(head(d,10))

apriori(d)

# set better support and confidence levels to learn more rules
transaction_rules <- apriori(d, parameter = list(support =0.002, confidence =0.20, maxlen = 3))
summary(transaction_rules)
plot(head(d,10))

# look at the first three rules
inspect(transaction_rules[1:2])

# sorting  descending transactions rules by lift to determine actionable rules
top.lift1 <- sort(transaction_rules, decreasing = TRUE, na.last = NA, by = "lift")
inspect(top.lift[1:5])



transaction_rules2 <- apriori(d, parameter = list(support =0.002, confidence =0.20, maxlen = 2))
inspect(transaction_rules2[1:5])

t1<-inspect(top.lift[1])
t2<-inspect(transaction_rules2[1])   

