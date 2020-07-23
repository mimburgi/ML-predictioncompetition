require(caret)
require(caretEnsemble)
require(dplyr)
require(ada)
require(plyr)
require(C50)
require(party)
require(gbm)
require(hda)
require(HDclassif)
require(MASS)
require(naivebayes)
require(nnet)
require(nodeHarvest)
require(e1071)
require(randomForest)
require(foreach)
require(import)
require(rpart)
require(RRF)
require(kernlab)
require(xgboost)

load("class_data.RData")

x$resp<-as.factor(y)
levels(x$resp)<-c("no", "yes") #caret requires names for classes
dat<-x


# set k and reps for tuning base and meta models####

baseK=10
baseReps=3
metaK=10
metaReps=3

# define base models####

#names need to be able to be passed to the method arg of train()
basemodels_nopreproc<-c(
  "knn",
  "xgbLinear",
  #"xyf", #todo: figure out if xyf is a good idea
  "xgbTree",
  "RRF",
  "parRF",
  "nodeHarvest",
  "ada",
  "C5.0",
  'naive_bayes', 
  "gbm",
  "rf",
  "cforest",
  "rpart",
  "hdda"
)
basemodels_pca<-c(
  "lda", 
  "nnet"
)

basemodels<-c(basemodels_nopreproc, basemodels_pca)

# define parameter grids for tuning base models####
knngrid<-expand.grid(k=seq(5, 25, 2))
nnetgrid<-expand.grid(size=seq(1, 11, 2),
                      decay=c(.01, .05, .1, .3, .5, .75))
rpartgrid<-expand.grid(cp=seq(.01, .1, .03))
cforestgrid<-expand.grid(mtry=c(2,seq(50, 350, 75), 450))
rfgrid<-expand.grid(mtry=c(2,seq(50, 350, 75), 450))
gbmgrid<-expand.grid(interaction.depth=seq(1, 13, 3), n.trees=seq(50, 125, 250), 
                    shrinkage=c(.1, .01), n.minobsinnode=10)
c5.0grid<-expand.grid(model=c("rules"), winnow=F, trials=c(20, 50, 75, 100))
adagrid<-expand.grid(iter=50, nu = .01, maxdepth=seq(3, 13, 3))
nodeHarvestgrid<-expand.grid(mode="mean", maxinter=c(2,4,6,8))
parRFgrid<-expand.grid(mtry=seq(100, 375, 75))
RRFgrid<-expand.grid(coefReg=c(.01,.5,1), 
                     coefImp=c(0, .5,1), 
                     mtry=c(2,seq(50, 350, 75), 450))
# xyfgrid<-expand.grid(xdim=c(2:9),
#                      ydim=c(2:9),
#                      user.weights=c(.2, .5, .7, .8, .9),
#                      topo=c("hexagonal"))
hddagrid<-expand.grid(threshold=c(.7, .8, .9, .93, .97, .999),
                      model="all")
xgbTreegrid<-expand.grid(gamma=0, min_child_weight=1,
                         eta=c(.1,.2,.3), colsample_bytree=c(.6, .8),
                         subsample=c(.5, .75, 1), nrounds=seq(50,125,250),
                         max_depth=c(2,seq(50, 350, 75), 450))
# define parameter grids for tuning meta model####
metannet_grid<-expand.grid(size=c(1,2,3),
                                  decay=c(0,.01,.1,.5,.75))

# tune base models####

set.seed(3)
baseFolds<-createMultiFolds(dat$resp, k=baseK, times=baseReps)
basecontrol=trainControl(method="repeatedcv", number=baseK, repeats=baseReps, classProbs = T,
                         savePredictions = "all",#necessary for meta model
                         index=baseFolds, #for reproducability
                         returnResamp = "final")      
      
# separate x and resp from metaTuneTrain so that caret can preprocess x
datx<-dat[,c(1:length(dat)-1)]
datresp<-dat$resp

#tune models that require no preproc using trControl in train()
for (model in basemodels_nopreproc){
  #check for a custom tuning grid
  if (exists((paste0(model, "grid")))){ #check for custom grid
    tuneGrid=get(paste0(model, "grid"))
    set.seed(23)
    fit<-train(datx, datresp, method=model, trControl = basecontrol,
                   tuneGrid=tuneGrid)
  }else{ #models with no custom grid
    fit<-train(datx, datresp, method=model, trControl = basecontrol)
  }
  #assign fit to unique variable
  assign(paste0("fit.", model), fit)
  write(paste(model, "finished\n"), "finalmodelprogress.txt", append = T)
}# end for loop for no preproc models
      

#preprocess for nnet, lda
pcatransform<-preProcess(datx, method=c("center", "scale", "pca"))
datx<-predict(pcatransform, datx)


#now tune those models that used pca preproc
set.seed(23)
fit.nnet<-train(datx_pca, datresp, method="nnet", trControl = basecontrol,
               tuneGrid=nnetgrid, MaxNWts=4000)
set.seed(23)
fit.lda<-train(datx_pca, datresp, method="lda", trControl = basecontrol)

# concat all base models to a single list
for (model in basemodels){
  fit<-get(paste0("fit.", model))
  fitlist<-as.list(fit)
  if(model==basemodels[1]){metainput<-fitlist}
  else{metainput<-as.list(c(metainput, fitlist))}
}

#convert class to caretList so caretStack will read it
class(metainput)<-"caretList"
      
#tune meta model ####

set.seed(51)
metaFolds<-createMultiFolds(dat$resp, k=metaK, times=metaReps)
metacontrol=trainControl(method="repeatedcv", number=metaK, repeats=metaReps, classProbs = T,
                         index=metaFolds) #for reproducability
set.seed(3)
final.fit<-caretStack(metainput, method="nnet", tuneGrid=metannet_grid, trControl=metacontrol)

#predict on xnew ####
xnew_pca<-predict(pcatransform, xnew)
xnew_withpca<-cbind(xnew, xnew_pca)
ynew<-predict(final.fit, xnew)
ynew<-ifelse(ynew=='yes', 1, 0) #convert back to original labels

#import accuracy estimates from nested CVs ####
nestedCV1<-read.table("avgMetaAccs_10CV2-shortcut2.txt", header = T)
nestedCV2<-read.table("avgMetaAccs_10CV3-shortcut2.txt", header = T)
nestedCV3<-read.table("avgMetaAccs_10CV4-shortcut2.txt", header = T)
test_acc<-mean(c(nestedCV1$as.numeric.acc.[nestedCV1$model=='nnet'],
                   nestedCV2$as.numeric.acc.[nestedCV1$model=='nnet'],
                   nestedCV3$as.numeric.acc.[nestedCV1$model=='nnet']))
test_error<-1-test_acc

save(ynew, test_error, file='925008899.RData')
