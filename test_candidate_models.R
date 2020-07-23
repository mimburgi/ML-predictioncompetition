#this was originally three scripts, one to test PCA preproc,
#one for the nested CV without stacking,
#one for nested CV for stacking
#each section has a header and the workspace is cleared to make sure no variables get carried over


##########################
##testing PCA preproc#####
##########################

library(caret)
load("class_data.RData")

###specify number of components to test####
PCAvals<-seq(220, 270, by=5)

####gbm testing at different component numbers####

gbmControl<-trainControl(method = "cv", number = 10) #10-fold CV

#need to initialize a list of results here to add to
gbmresults<-data.frame(
  components=PCAvals,
  predAcc=rep(NA, length(PCAvals))
)


#then loop through component numbers and add to those results
for (i in 1:(length(PCAvals)-1)){ #don't include max number of components(no PCA)
  components<-PCAvals[i]
  preprocess_x <- preProcess(x, method=c("center", "scale", "pca"), pcaComp = components)
  trimmedx<-predict(preprocess_x, x)
  dat<-trimmedx
  dat$resp<-as.factor(y)
  set.seed(23)
  fit.gbm<-train(resp~., data = dat, method = "gbm",metric = "Accuracy", trControl = gbmControl)
  gbmresults$predAcc[i]<-max(fit.gbm$results$Accuracy)
}
#add result for no preprocessing
dat<-x
dat$resp<-as.factor(y)
set.seed(23)
fit.gbm<-train(resp~., data = dat, method = "gbm",metric = "Accuracy", trControl = gbmControl)
gbmresults$predAcc[length(PCAvals)]<-max(fit.gbm$results$Accuracy)


####rf testing at different component numbers####

rfControl<-trainControl(method = "cv", number = 10) #10-fold CV

#need to initialize a list of results here to add to
rfresults<-data.frame(
  components=PCAvals,
  predAcc=rep(NA, length(PCAvals))
)


#then loop through component numbers and add to those results
for (i in 1:(length(PCAvals)-1)){ #don't include max number of components(no PCA)
  components<-PCAvals[i]
  preprocess_x <- preProcess(x, method=c("center", "scale", "pca"), pcaComp = components)
  trimmedx<-predict(preprocess_x, x)
  dat<-trimmedx
  dat$resp<-as.factor(y)
  set.seed(23)
  fit.rf<-train(resp~., data = dat, method = "rf",metric = "Accuracy", trControl = rfControl)
  rfresults$predAcc[i]<-max(fit.rf$results$Accuracy)
}
#add result for no preprocessing
dat<-x
dat$resp<-as.factor(y)
set.seed(23)
fit.rf<-train(resp~., data = dat, method = "rf",metric = "Accuracy", trControl = rfControl)
rfresults$predAcc[length(PCAvals)]<-max(fit.rf$results$Accuracy)


###########################
##nested CV no stacking####
###########################
rm(list=ls())

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
levels(x$resp)<-c("no", "yes")
dat<-x

# set k and reps for each cv####

#will want to change length of seed lists if any of these are changed
#if more seeds are needed than are provided, it should go back to the beginning of the seed lists
#todo: test that safeguard
outerK=10
outerReps=3
middleK=10
middleReps=1
innerK=10
innerReps=1

# define base models####

#names need to be able to be passed to the method arg of train()
basemodels<-c(
  #"knn",
  #"xgbLinear",
  #"xyf", #todo: figure out if xyf is a good idea
  #"xgbTree",
  "rpart",
  "RRF",
  "parRF",
  "nodeHarvest",
  #"ada",
  "C5.0",
  #'naive_bayes', 
  "gbm",
  "rf",
  "cforest",
  "hdda"
)

# define parameter grids for tuning models as needed####

knngrid<-expand.grid(k=seq(5, 30, 2))
nnetgrid<-expand.grid(size=seq(1, 25, 2),
                      decay=c(.01, .05, .1, .5, .75))
rpartgrid<-expand.grid(cp=seq(.01, .11, .02))
#cforestgrid<-expand.grid(mtry=c(2, 20 ,seq(100, 500, 100)))
rfgrid<-expand.grid(mtry=c(2, 20 ,seq(100, 500, 100)))
gbmgrid<-expand.grid(interaction.depth=seq(1, 13, 4), n.trees=c(50, 250, 100), 
                     shrinkage=c(.1, .01), n.minobsinnode=10)
#c5.0grid<-expand.grid(model=c("rules"), trials=c(20, 50, 80, 100))
#adagrid<-expand.grid(iter=c(50, 100, 150), nu = .01, maxdepth=seq(3, 13, 2))
nodeHarvestgrid<-expand.grid(mode="mean", maxinter=c(1,3,4,6,8))
parRFgrid<-expand.grid(mtry=seq(100, 400, 100))
RRFgrid<-expand.grid(coefReg=c(.01,.5),
                     coefImp=c(0, .5),
                     mtry=c(20, 250, 400))


# create seed lists####

#to generate folds for Inner CV
#gets done 30 times (10 outer folds times 3)
innerFoldseeds<-c(46, 442, 798, 122, 559, 206, 127, 749, 888, 372, 659, 94, 380, 271, 804, 999, 998, 799, 781, 432, 740, 617, 695, 1, 464, 215, 370, 597, 342, 108)
innerFoldseed_ctr=1 # to index middleFoldseeds, will get added to every time this is used

# create folds for outer cv####
set.seed(920)
outerFolds<-createMultiFolds(dat$resp, k = outerK, times = outerReps)

# begin loop for repititions of outer CV####
for (outerRep in 1:outerReps){
  # begin loop for outer cv  (testing accuracy of tuned meta models)####
  for (i in 1:outerK){ #begin outer cv
    twodigit_foldnum<-ifelse(i < 10, paste0(0, i), i)
    Acc_indeces<-outerFolds[[paste0("Fold", twodigit_foldnum, ".Rep", outerRep)]]
    #set training and testing set
    AccTrain<-dat[c(Acc_indeces),]
    AccTest<-dat[-c(Acc_indeces),]
    
    set.seed(innerFoldseeds[innerFoldseed_ctr])
    innerFoldseed_ctr=innerFoldseed_ctr+1
    baseTuneFolds<-createFolds(AccTrain$resp, k=innerK, returnTrain = T)
    
    # tune models
    basecontrol=trainControl(method="cv", number=innerK, classProbs = T,
                               index=baseTuneFolds)
    # start inner CV (tuning base models, done internally within train)####
      
      
    # separate x and resp from AccTrain so that caret can preprocess x
    AccTrainx<-AccTrain[,c(1:length(AccTrain)-1)]
    AccTrainresp<-AccTrain$resp
      
      #tune models using trControl in train()
    for (model in basemodels){
    start<-Sys.time()
        #check for a custom tuning grid
      if (exists((paste0(model, "grid")))){ #check for custom grid
        tuneGrid=get(paste0(model, "grid"))
        fit<-train(AccTrainx, AccTrainresp, method=model, trControl = basecontrol,
                     tuneGrid=tuneGrid)
      }else{ #models with no custom grid
        fit<-train(AccTrainx, AccTrainresp, method=model, trControl = basecontrol)
      }
      #assign fit to unique variable
      assign(paste0("fit.", model), fit)
    end<-Sys.time()
    print(model)
    print(end-start)
    }# end for loop for no basemodels
      
      
    
    
    #initialize df to hold all accuracy estimates across each outer fold
    #only done once on first iteration of the whole algorithm
    if (i==1 & outerRep==1){all_accs=data.frame(model=character(),
                                                 CVfold=numeric(),
                                                 Rep=numeric(),
                                                 acc=numeric(),
                                                 stringsAsFactors = F)}
    
    # separate x and resp for more readable code below
    AccTestx<-dplyr::select(AccTest,-resp) #remove last column (resp)
    AccTestresp<-AccTest$resp

    
    
    #estimate accuracy for each meta model
    for (model in basemodels){
      fit<-get(paste0("fit.",model))
      preds<-predict(fit, AccTestx)
      acc<-sum(preds==AccTestresp)/length(AccTestresp)
      all_accs[nrow(all_accs)+1, ] <- c(model, i, outerRep, acc)
    }

    
  }#end loop for outer cv
}#end outer cv reps


#print summary of average accuracy for each meta model####
avgAccs<-aggregate(as.numeric(acc) ~ model, data=all_accs, FUN=mean)


#############################
##nested CV with stacking####
#############################
rm(list=ls())

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
levels(x$resp)<-c("no", "yes")
dat<-x

# define calcMetaAcc function####
calcMetaAcc <- function(caretList, method, tuneGrid=NULL, 
                        trControl=trainControl(method="none"), 
                        xnew=xnew, respnew=respnew){
  fit<-caretStack(caretList, method=method, 
                  trControl=trControl, tuneGrid=tuneGrid)
  preds<-predict(fit, xnew)
  acc<-sum(preds==respnew)/length(respnew)
  as.numeric(acc)
}

# set k and reps for each cv####

#will want to change length of seed lists if any of these are changed
outerK=10
outerReps=1
middleK=5
middleReps=1
innerK=5
innerReps=1

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
# define meta models####

#these names need to be able to be passed to the method arg of train
metamodels_forTuning<-c("nnet", "rf", "svmPoly", "gbm", "knn", "RRF") 
metamodels_noTuning<-c("glm", "lda") #meta models for which there are no pars to tune
metamodels<-c(metamodels_forTuning, metamodels_noTuning)
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
# define parameter grids for tuning meta models####
metannet_grid<-expand.grid(size=c(1,2,3),
                           decay=c(0,.01,.1,.5,.75))
metarf_grid<-expand.grid(mtry=c(1,2,5,9,17))
metasvmPoly_grid<-expand.grid(degree=c(1:3),
                              scale=c(.001, .01, .05, .1),
                              C=c(.25, .5, 1))
metaknn_grid<-expand.grid(k=seq(3,11,2))
metaRRF_grid<-expand.grid(mtry=c(1,2,5,9,17),
                          coefReg=c(.01, .5, 1),
                          coefImp=c(0,.5,1))
metagbm_grid<-expand.grid(interaction.depth=c(1:4),
                          n.trees=seq(50,425,75),
                          shrinkage=c(.01, .1),
                          n.minobsinnode=10)

# create seed lists####

#to generate folds for Middle CV
#gets done 10 times (10 outer folds)
middleFoldseeds<-c(233, 584, 816, 647, 71, 963, 815, 378, 553, 746)
middleFoldseed_ctr=1 # to index middleFoldseeds, will get added to every time this is used

#to generate folds for Inner CV
#gets done 10 times (10 outer folds) times times 10 middle CV so 100 seeds
innerFoldseeds<-c(637, 554, 351, 183, 18, 969, 40, 939, 685, 105, 155, 
                  990, 110, 291, 419, 631, 822, 162, 1, 319, 97, 498, 
                  71, 464, 951, 167, 829, 492, 115, 792, 359, 58, 809, 
                  443, 871, 100, 518, 794, 714, 213, 269, 433, 119, 834, 
                  734, 642, 70, 777, 865, 265, 721, 675, 958, 26, 108, 680, 
                  140, 778, 330, 888, 791, 456, 204, 252, 885, 882, 197, 43, 
                  671, 416, 526, 438, 242, 273, 410, 284, 776, 613, 534, 380, 
                  957, 690, 587, 920, 641, 976, 655, 211, 79, 260, 2, 471, 
                  495, 805, 474, 937, 555, 808, 560, 827)
innerFoldseed_ctr=1 # to index innerFoldseeds, will get added to every time this is used

# create folds for outer cv####
set.seed(410)
outerFolds<-createMultiFolds(dat$resp, k = outerK, times = outerReps)

# begin loop for repititions of outer CV####
for (outerRep in 1:outerReps){
  # begin loop for outer cv  (testing accuracy of tuned meta models)####
  for (i in 1:outerK){ #begin outer cv
    twodigit_foldnum<-ifelse(i < 10, paste0(0, i), i)
    metaAcc_indeces<-outerFolds[[paste0("Fold", twodigit_foldnum, ".Rep", outerRep)]]
    #set training and testing set
    metaAccTrain<-dat[c(metaAcc_indeces),]
    metaAccTest<-dat[-c(metaAcc_indeces),]
    
    #set folds for middle cv
    set.seed(middleFoldseeds[middleFoldseed_ctr])
    metaTuneFolds<-createFolds(metaAccTrain$resp, k=middleK, returnTrain = T)
    middleFoldseed_ctr=middleFoldseed_ctr+1
    if(middleFoldseed_ctr > length(middleFoldseeds)){middleFoldseed_ctr=1} #in case i counted wrong
    
    # begin loop for middle cv (tuning meta models)####
    for (j in 1:middleK){ #begin middle cv
      metaTune_indeces<-metaTuneFolds[[j]]
      #set middle training and testing set
      metaTuneTrain<-metaAccTrain[c(metaTune_indeces),]
      metaTuneTest<-metaAccTrain[-c(metaTune_indeces),]
      
      #set folds and trainControl for inner CV
      set.seed(innerFoldseeds[innerFoldseed_ctr])
      baseTuneFolds<-createFolds(metaTuneTrain$resp, k=innerK, returnTrain = TRUE)
      innerFoldseeds_ctr=innerFoldseed_ctr+1
      if(innerFoldseed_ctr > length(innerFoldseeds)){innerFoldseed_ctr=1} #in case i counted wrong
      
      basecontrol=trainControl(method="cv", number=innerK, classProbs = T,
                               savePredictions = "all",#necessary for middle models
                               index=baseTuneFolds, #for reproducability
                               returnResamp = "final")
      # start inner CV (tuning base models, done internally within train)####
      
      
      # separate x and resp from metaTuneTrain so that caret can preprocess x
      metaTuneTrainx<-metaTuneTrain[,c(1:length(metaTuneTrain)-1)]
      metaTuneTrainresp<-metaTuneTrain$resp
      
      #tune models that require no preproc using trControl in train()
      for (model in basemodels_nopreproc){
        #check for a custom tuning grid
        if (exists((paste0(model, "grid")))){ #check for custom grid
          tuneGrid=get(paste0(model, "grid"))
          fit<-train(metaTuneTrainx, metaTuneTrainresp, method=model, trControl = basecontrol,
                     tuneGrid=tuneGrid)
        }else{ #models with no custom grid
          fit<-train(metaTuneTrainx, metaTuneTrainresp, method=model, trControl = basecontrol)
        }
        #assign fit to unique variable
        assign(paste0("fit.", model), fit)
      }# end for loop for no preproc models
      
      
      #preprocess for nnet, lda
      pcatransform<-preProcess(metaTuneTrainx, method=c("center", "scale", "pca"))
      metaTuneTrainx_pca<-predict(pcatransform, metaTuneTrainx)
      
      
      #now tune those models that used pca preproc
      fit.nnet<-train(metaTuneTrainx_pca, metaTuneTrainresp, method="nnet", trControl = basecontrol,
                      tuneGrid=nnetgrid, MaxNWts=4000)
      fit.lda<-train(metaTuneTrainx_pca, metaTuneTrainresp, method="lda", trControl = basecontrol)
      
      # concat all base models to a single list
      for (model in basemodels){
        fit<-get(paste0("fit.", model))
        fitlist<-as.list(fit)
        if(model==basemodels_nopreproc[1]){metainput<-fitlist}
        else{metainput<-as.list(c(metainput, fitlist))}
      }
      
      #convert class to caretList so caretStack will read it
      class(metainput)<-"caretList"
      
      #tune meta models on middle training fold####
      
      # separate x and resp for metaTuneTest for more readable code below
      metaTuneTestx<-dplyr::select(metaTuneTest,-resp) #remove last column (resp)
      metaTuneTestresp<-metaTuneTest$resp
      
      #add pca decomp to testing set so lda and nnet can make predictions
      metaTuneTest_pca<-predict(pcatransform, metaTuneTestx) #decomp in the same way that the training set was
      metaTuneTestx<-cbind(metaTuneTestx, metaTuneTest_pca) #add to non transformed data
      
      #tuning meta models across grids begins here
      #stores accuracy predictions across each middle CV fold in model_outgrid
      #ex. nnet_outgrid contains accuracies for each fold for each par combo in the nnet meta
      for (model in metamodels_forTuning){
        modelGrid<-get(paste0("meta",model,"_grid")) #get grid for this model
        if (j==1){#if this is the first fold of the middle CV, initialize outgrid
          #uses tuning grid as a base because it contains parameter names
          assign(paste0(model,"_outgrid"), modelGrid)
        }
        for (parcombo_number in 1:nrow(modelGrid)){
          parcombo_acc<-calcMetaAcc(metainput, method=model, tuneGrid=modelGrid[parcombo_number,,drop=F],
                                    xnew=metaTuneTestx, respnew = metaTuneTestresp)
          
          #todo: clean these next few lines up, there has to be a better way 
          #assign existing outgrid to tmpgrid
          tmpgrid<-get(paste0(model,"_outgrid"))
          
          #initializing CVacc as NAs because without this step, every value in the column gets assigned the first accuracy, then overwritten
          #safer to initialize as NAs; if something goes wrong, it's obvious 
          if (parcombo_number==1){tmpgrid[[paste0("accCV", j)]]<-NA}
          
          #add accuracy for this parameter combo to the dataframe
          tmpgrid[[paste0("accCV", j)]][parcombo_number]<-parcombo_acc
          #replace model outgrid with tmpgrid
          assign(paste0(model,"_outgrid"), tmpgrid)
          
        } #end loop through parameter combos in grid
      } #end loop across meta models
    }#end loop for middle cv
    
    #average accuracy across middle CV iterations
    #stored in an avgAcc column in the model outgrid
    for (model in metamodels_forTuning){
      modelresults<-get(paste0(model, "_outgrid"))
      CVaccs<-dplyr::select(modelresults, contains('CV'))
      modelresults$avgAcc<-rowMeans(CVaccs)
      assign(paste0(model, "_outgrid"), modelresults)
    }
    
    #get best pars for each meta model to predict accuracy with in next step
    #stored in modelMetaPars (ex nnetMetaPars)
    for (model in metamodels_forTuning){
      modelresults<-get(paste0(model, "_outgrid"))
      bestpars<-modelresults[(modelresults$avgAcc==max(modelresults$avgAcc)),] 
      bestpars<-dplyr::select(bestpars,  -contains("acc")) #retain only pars, not accs
      if (nrow(bestpars) > 1){ #in case of a tie, pick the top one
        bestpars<-bestpars[1,,drop=F] #drop=F to ensure names are retained
      }
      assign(paste0(model, "MetaPars"), bestpars)
    }
    
    #initialize df to hold all meta accuracy estimates across each outer fold
    #only done once on first iteration of the whole algorithm
    if (i==1 & outerReps==1){all_accs=data.frame(model=character(),
                                                 CVfold=numeric(),
                                                 Rep=numeric(),
                                                 acc=numeric(),
                                                 stringsAsFactors = F)}
    
    # separate x and resp for metaAccTest for more readable code below
    metaAccTestx<-dplyr::select(metaAccTest, -resp) #remove resp
    metaAccTestresp<-metaAccTest$resp
    #need to add pca decomp to testing set so lda and nnet base models can make predictions
    metaAccTest_pca<-predict(pcatransform, metaAccTestx) #decomp in the same way that the training set was
    metaAccTestx<-cbind(metaAccTestx, metaAccTest_pca) #add to non transformed data
    
    #estimate accuracy for each meta model
    for (model in metamodels_forTuning){
      modelGrid<-get(paste0(model,"MetaPars"))
      acc<-calcMetaAcc(metainput, method=model, tuneGrid=modelGrid, 
                       xnew=metaAccTestx, respnew = metaAccTestresp)
      all_accs[nrow(all_accs)+1, ] <- c(model, i, outerRep, acc)
    }
    for (model in metamodels_noTuning){
      acc<-calcMetaAcc(metainput, method=model, xnew=metaAccTestx, respnew = metaAccTestresp)
      all_accs[nrow(all_accs)+1, ] <- c(model, i, outerRep, acc)
    }
    
  }#end loop for outer cv
}#end outer cv reps