
# MuliClass classification with XGBoost


library("xgboost")  # the main algorithm
library("archdata") # for the sample dataset
library("caret")    # for the confusionmatrix() function (also needs e1071 package)
library("e1071")
library("doSNOW")
library("kernlab")
library("ranger")
library("C50")
library("caTools")
library("plyr")
library("caretEnsemble")

# DATASET: 
# Concentrations for eleven trace elements in waste glass specimens for three archaeology
#furnance sites in England
# Modified dataset from R package "archdata"; specifically data(RBGlass2) 
# Romano-British Glass, Trace Elements
# Data Frame with 164 observations of 12 variables



#Site: Three factors represeting furnancies at three differnt geographic locations
#Ba: Barium ppm
#Co: Cobalt ppm
#Cr: Chromium ppm
#Cu: Copper ppm
#Li: Lithium ppm
#Ni: Nickel ppm
#Sr: Strontium ppm
#V: Vanadium ppm
#Y: Yttriu ppm
#Zn: Zinc ppm
#Zr: Zirconium ppm




dat <- read.csv("C://R Local Repository//Caret_XGBoost//ArchData.csv", stringsAsFactors=FALSE)
dat$X<-NULL
dat$Site<-as.factor(dat$Site)


# Create balanced stratified train and test indices
indexes <- createDataPartition(dat$Site,times = 1,p = 0.7,list = FALSE)

Location.train <- dat[indexes,]
Location.test <- dat[-indexes,]

#Verify the proportions are the same
prop.table(table(dat$Site))
prop.table(table(Location.train$Site))
prop.table(table(Location.test$Site))


#Set up for repeated cross validations

train.control <- trainControl(method = "repeatedcv",
                              summaryFunction = multiClassSummary,
                              number =8,
                              repeats = 3,
                              search = "grid")

#Let caret adjust the xgboost paramters automatically

tune.grid <- expand.grid(eta = c(0.03, 0.040, 0.2),
                         nrounds = c(30, 60, 90),
                         max_depth = 3:9,
                         min_child_weight = c(1.0, 2.5, 3.75),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)


# Create parallel processes for modeling
# Running a i7-7820HQ Intel Processor with 4 cores...so make 4 Clusters
cl <- makeCluster(4, type = "SOCK")



# Register cluster so that caret will know to train in parallel.

registerDoSNOW(cl)

#Train the model with training data and let caret return the best model
caret.cv <- train(Site ~ ., 
                  data = Location.train,
                  method = "xgbTree",
                  metric="Accuracy",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)

#Look at the predictions and the associated confustion matrix
preds <- predict(caret.cv, Location.test)
confusionMatrix(preds, Location.test$Site)


plot(caret.cv)


#######################################################################################
# Try LogitBoost, XGBoost,   and Random Forest to see how their performance compares
######################################################################################



control <- trainControl(method="repeatedcv",
                        number=8,
                        repeats=3,
                        savePredictions=TRUE)

ensembleModelList<-c('LogitBoost','xgbTree','C5.0','ranger')
set.seed(12345)

cl <- makeCluster(4, type = "SOCK")

ensembleModel<-caretList(Site~.,
                  data = Location.train,
                  trControl=control,
                  methodList=ensembleModelList)
        

                        
stopCluster(cl)

results<-(resamples(ensembleModel))
summary(results)
dotplot(results)

