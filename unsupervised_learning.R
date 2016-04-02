#!/usr/bin/env Rscript

###########################################################################
## Chris Hodapp (chodapp3)
## Georgia Institute of Technology
## CS7641, Machine Learning, Spring 2016
## Assignment 3, Unsupervised Learning (2016-04-03)
###########################################################################

library(doParallel);
registerDoParallel(4);
library(ggplot2);
library(RSNNS);
library(mclust);
library(cluster);
library(fastICA);
library(RPEnsemble);
library(FSelector);

source("multiplot.R");

###########################################################################
## General
###########################################################################

## Split ratio 'f' for training data, and the rest (1-f) for testing.
## Returns a list with items "train" and "test" containing rows taken,
## in random order, from "frame"; list also has item "idx" which is
## the indices from 'frame' that were used for the training set (and
## all others were used for the testing set).
splitTrainingTest <- function(frame, f) {
    trainIdx <- sample(nrow(frame), size=f*nrow(frame));
    train <- frame[trainIdx,];
    test <- frame[-trainIdx,];
    return(list(train = train, test = test, idx = trainIdx));
}

## Turn a confusion matrix into an error rate.  It doesn't matter
## whether targets are on rows and predictions are on columns, or vice
## versa.
confusionToError <- function(mtx) {
    return(1 - sum(diag(mtx)) / sum(mtx));
}

###########################################################################
## Data Loading & Other Boilerplate
###########################################################################

print("Loading & converting data...");
# pdf("assignment1_plots.pdf");

## Load data for "Steel Plates Faults" & apply proper header names:
faults <- read.table("Faults.NNA", sep="\t", header=FALSE);

## Remove a pesky outlier:
faults <- faults[-392,];

colnames(faults) <- readLines("Faults27x7_var");
## If these are missing, use:
## https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults
## https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA:
## https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults27x7_var

## Dependent variables are several exclusive binary columns:
depCol <- c("Pastry", "Z_Scratch", "K_Scatch", "Stains",
            "Dirtiness", "Bumps", "Other_Faults");

## Also standardize data to mean 0, variance 1, separating labels:
faultsNorm <- data.frame(scale(faults[-which(names(faults) %in% depCol)]))
faultsLabels <- faults[depCol];
inputNames <- colnames(faultsNorm);
labelNames <- colnames(faultsLabels);

## Turn fault labels into a factor (may need this someplace):
faultsFactor <- factor(apply(faultsLabels, 1, function(x) which(x == 1)),
                       labels = depCol);

## Split training & test:
tt <- splitTrainingTest(cbind(faultsNorm, faultsLabels), 0.8);
faultsTrain <- tt$train;
faultsTest <- tt$test;

## Load data for "Letter Recognition" data set & apply headers:
letters <- read.table("letter-recognition.data", sep=",", header=FALSE);
colnames(letters) <- c("Letter", "Xbox", "Ybox", "Width", "Height",
                       "OnPix", "Xbar", "Ybar", "X2bar", "Y2bar",
                       "XYbar", "X2Ybar", "XY2bar", "Xedge",
                       "XedgeXY", "Yedge", "YedgeYX");
## https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

## Standardize this too:
lettersNorm <- data.frame(
    scale(letters[-which(names(letters) == "Letter")]));
## And turn 'Letter' into a 26-wide binary matrix (columns A-Z):
lettersLabels <- as.data.frame(model.matrix( ~ 0 + Letter, letters));
colnames(lettersLabels) <- levels(letters$Letter);

###########################################################################
## k-means
###########################################################################

## Given clusters produced with 'kmeans', any number of instances, and
## a corresponding list of cluster indices for each instance, return a
## cumulative histogram giving the probability that a point of that
## cluster is at that distance, or less.  This will be a 'stacked'
## histogram - the data frame will have column 'class' which gives the
## class's index as an integer, column 'hist' which contains the
## cumulative probability, and column 'bins' which gives the midpoint
## of the respective bin for that probability.
distHistogram <- function(clusters, data, idxs) {
    ## Compute squared distance from each instance to the cluster that
    ## 'owns' it:
    sqDist <- data.frame(
        sqDist = rowSums(sqrt((clusters$center[idxs,] - data)^2)),
        class = idxs);
    sqDistSum <- aggregate(sqDist ~ class, sqDist, sum);
    bins <- 500;
    ## In 'breaks' we must include the max(...) or otherwise 'hist'
    ## won't complete, as values will fall outside the bins.  We set
    ## the rest to 2*mean because that seems (empirically) to be a
    ## useful range, and then remove the final bin at the end because
    ## it distorts the axis.
    breaks <- c(seq(0,
                    mean(sqDist$sqDist) * 2,
                    length=bins),
                max(sqDist$sqDist));

    sqDistHist <- aggregate(
        sqDist ~ class,
        sqDist,
        function(x) {
            h <- hist(x, breaks = breaks, plot = FALSE);
            ## Normalize the densities to the bin sizes, and accumulate:
            return(cumsum(h$density * diff(breaks)));
        },
        simplify = FALSE);
    ## This just passes dummy data to get the midpoints:
    mids <- hist(breaks, breaks = breaks, plot = FALSE)$mids;
    dtmp <- apply(
        sqDistHist, 1,
        function(df) data.frame(
                         ## Drop the last bin (it's just a catch-all
                         ## and distorts axes):
                         hist = df$sqDist[-bins],
                         class = df$class,
                         bins = mids[-bins]));
    distFlat <- do.call(rbind, dtmp);
    return(distFlat);
}

## Given clusters produced by 'kmeans' and a set of labels
## corresponding to the data used to generate the clusters, performs
## some prediction by treating each cluster as producing the average
## of the labels that are in it.  This returns a data frame with a
## column 'class' for the class index, 'argmax' for a factor that
## corresponds to the identified label (the argmax of that average),
## 'errRate' for a ratio of incorrect labels in that cluster, 'err'
## for an actual number of incorrectly-labeled instances, and 'size'
## for the size of the cluster.
clusterPredictErr <- function(clusters, labels) {
    cols <- colnames(labels);
    ## Put classes alongside labels so we can aggregate:
    labels$class <- clusters$cluster;
    ## Average the labels (as binary vectors) across each class:
    labelsAvg <- aggregate(. ~ class, labels, mean);
    ## Set 'argmax' to the factor that has the highest average: (sort
    ## of like we did with the neural networks)
    labelsAvg$argmax <- factor(apply(labelsAvg[cols], 1, which.max),
                               labels = cols, levels = 1:length(cols));
    ## Set 'errRate' to the sum of other factors (these are all
    ## "wrong" if we use the highest factor):
    labelsAvg$errRate <- apply(
        labelsAvg[cols], 1, function(x) (1 - max(x)));
    ## Then set 'err' to that multiplied by the cluster size to tell
    ## us error as a number of instances, not as a rate:
    labelsAvg$size <- clusters$size[labelsAvg$class];
    labelsAvg$err <- labelsAvg$errRate * labelsAvg$size;
    return(labelsAvg);
}

## Given clusters produced by 'kmeans' and a set of labels
## corresponding to the data used to generate the clusters, treats
## each cluster as having the label of whatever label occurs most
## often in its instances, and produces a confusion matrix from it.
## The column headings refer to the correct label, and the row
## headings refer to the 'identified' label.
clusterConfusionMatrix <- function(clusters, labels) {
    cols <- colnames(labels);
    ## Put classes alongside labels so we can aggregate:
    labels$class <- clusters$cluster;
    ## Tally up the correct number of each letter in the cluster:
    labelsAvg <- aggregate(. ~ class, labels, sum);
    labelsAvg$class <- NULL;
    ## Figure out which label we should assign this cluster:
    labelsAvg$argmax <- factor(apply(labelsAvg[cols], 1, which.max),
                               labels = cols, levels = 1:length(cols));
    ## Tally up, for each cluster label, what the actual labels are:
    conf <- aggregate(. ~ argmax, labelsAvg, sum);
    ## TODO: Not just reimplement confusionMatrix above.
    rownames(conf) <- conf$argmax;
    conf$argmax <- NULL;
    return(as.matrix(conf));
};

## Given a confusion matrix (assuming that each column indicates a
## target - that is, that the row names refer to targets), return a
## data frame with column 'target' containing each target, 'nearest'
## containing the incorrect prediction that is most likely, and
## 'nearestRatio' giving how likely that incorrect prediction is.
getNearest <- function(confMtx) {
    d <- diag(confMtx);
    ## First, remove the main diagonal from the matrix, as that is the
    ## "correct" predictions:
    confWrong <- confMtx - diag(d);
    ## Find max & argmax within each column; this is the nearest
    ## incorrect prediction.
    nearestCount <- apply(confWrong, 2, max);
    nearestIdx <- apply(confWrong, 2, which.max);
    ## Reference 'nearestIdx' into the row names to get a name, and
    ## also divide count by column sums to get a ratio:
    return(data.frame(target = rownames(confWrong),
                      nearest = rownames(confWrong)[nearestIdx],
                      nearestRatio = nearestCount / colSums(confMtx)));
};

## This will take awhile (and isn't really needed except for
## getKmeansClusters):
getDissimMtx <- function() {
    ## Generate dissimilarity matrices for steel faults:
    tryCatch(
        faultsDissim <- local({
            load("faultsDissim.Rda");
            return(faultsDissim);
        }),
        error = function(w) {
            faultsDissim <- daisy(faultsNorm);
            save(faultsDissim, file="faultsDissim.Rda");
        });
    ## and likewise for letters (this will generate about 1.4 GB, beware):
    tryCatch(
        lettersDissim <- local({
            load("lettersDissim.Rda");
            return(lettersDissim);
        }),
        error = function(w) {
            lettersDissim <- daisy(lettersNorm);
            save(lettersDissim, file="lettersDissim.Rda");
        });
};

## Works, but plots aren't particularly useful:
## dissMtx <- dist(clusters$centers, upper = TRUE, diag = TRUE);
## clustersHist <- distHistogram(clusters, faultsNorm, clusters$cluster);
## ggplot(data=clustersHist,
##        aes(x=bins, y=hist, group=factor(class))) +
##     geom_line(aes(colour=factor(class))) +
##     xlab("Distance to cluster center") +
##     ylab("Cumulative probability") +
##     ggtitle("Distance distribution in each cluster")

## For a given range of k values and some data, returns a data frame
## with column "k" for k value and "withinSs" for the average
## within-cluster squared error.
getKmeansSSE <- function(data, ks, iters, runs) {
    foreach(k = ks, .combine='rbind') %dopar% {
        cat(k);
        cat("..");
        cl <- kmeans(data, k, iters, runs);
        return(data.frame(
            k = k,
            withinSs = cl$tot.withinss / nrow(data)));
    };
};

###########################################################################
## EM
###########################################################################


## For input 'mclust' (a model from the Mclust function) and 'labels'
## (a binary matrix corresponding to each input used for EM), returns
## a matrix which gives the probabilities of each cluster having the
## given class.  That matrix has one row per cluster, and one column
## per label.
emClassMtx <- function(mclust, labels) {
    ## First, get the Z matrix from the model.  The Z matrix has one
    ## row for each instance and one column for each cluster, each row
    ## giving the probabilities that that instance belongs to each
    ## cluster.
    z <- as.matrix(mclust$z);
    ## The labels are already the matrix we need: One row per
    ## instance, one column per class.  From this and Z, we may
    ## produce a matrix which gives the mean label of each cluster -
    ## one row per cluster, one column per class.
    clMean <- t(z) %*% as.matrix(labels);
    ## It will however need normalization:
    clMean <- clMean / rowSums(clMean);
    return(clMean);
}

## For input 'mclust' (a model from the Mclust function) and 'labels'
## (a binary matrix corresponding to each input used for EM), returns
## a matrix, same size as 'labels', which gives the probability of
## each corresponding label, according to the model.
emLabel <- function(mclust, labels) {
    z <- as.matrix(mclust$z);
    clMtx <- emClassMtx(mclust, labels);
    return(z %*% clMtx);
}

## For input 'mclust' (a model from the Mclust function) and 'labels'
## (a binary matrix corresponding to each input used for EM), returns
## a confusion matrix for each class - assuming that the model is used
## to predict the outcomes by the probabilities in each cluster.
emConfusion <- function(mclust, labels) {
    emLabels <- emLabel(mclust, labels);
    ## Is the below just encodeClassLabels?
    getFactor <- function(l) {
        factor(apply(l, 1, which.max),
               labels = colnames(labels),
               levels = 1:length(colnames(labels)))
    };
    correct <- getFactor(labels);
    predicted <- getFactor(emLabels);
    ## Transpose this to make it consistent with other spots:
    cm <- t(confusionMatrix(correct, predicted));
    ## and get proper row & column names (or just nothing):
    rownames(cm) <- colnames(labels);
    colnames(cm) <- colnames(labels);
    return(cm);
};

## Turn a model from the Mclust function into a data frame with
## columns "numClusters" giving the number of clusters from the model
## and "bic" giving the best BIC value at that each number of
## clusters.
emGetBic <- function(mclust) {
    ## and then find the max (which is a BIC) & argmax (which is a
    ## model, EII/VII/EEI/etc.) within each row:
    bic <- mclust$BIC;
    m <- apply(bic, 1, function(x) max(x, na.rm = TRUE));
    data.frame(
        bic = m / nrow(lettersNorm),
        numClusters = strtoi(rownames(bic)));
};

###########################################################################
## PCA
###########################################################################

## Given a PCA loading matrix and some (compatible) data, compute the
## reconstruction error from using just the 1st principal, the first 2
## principals, first 3, etc. If loading matrix has dimensions PxL,
## then data must have dimensions NxP.
reconstrError <- function(mtx, data) {
    foreach(dims=1:nrow(mtx), .combine='c') %dopar% {
        mtxR <- as.matrix(mtx[,1:dims]);
        scores <- as.matrix(data) %*% mtxR
        reconstr <- scores %*% t(mtxR)
        return(sum((data - reconstr)^2));
    }
}

## Given an object of class "prcomp" and a proportion of energy
## (default 0.90) to preserve of the total variance, returns the total
## number of dimensions required to preserve that amount of energy.
minDimension <- function(pca, energy = 0.90) {
    ## Find the lower limit of the variance:
    limit <- sum(pca$sdev^2) * energy;
    ## Figure out which indices *exceed* this:
    return(min(which(cumsum(pca$sdev^2) > limit)));
    ## Those are numbered from 1, so minimum is then all that we need.
}

###########################################################################
## fastICA
###########################################################################

## Given some data input (assumed to already be standardized) and a
## range of dimensions, performs ICA to the given range of dimensions,
## and returns reconstruction error (sum of squared error,
## particularly) as a data frame with columns "dims" for number of
## dimensions and "err" for reconstruction error.
icaReconstrError <- function(data, dimRange) {
    foreach(dims = dimRange, .combine = "rbind") %dopar% {
        ica <- fastICA(data, dims);
        A <- as.matrix(ica$A);
        S <- as.matrix(ica$S);
        return(data.frame(
            dims = dims,
            err = sum(((S %*% A) - data)^2)));
    };
}

###########################################################################
## Random projections
###########################################################################

## Rough look-alike to the MATLAB function; generate an m x n matrix
## with normally-distributed values of mean 0 & variance 1.
randn <- function(m, n) matrix(rnorm(m*n), m, n);

## Compute pseudoinverse with SVD.
pinv <- function(mtx) {
    s <- svd(mtx);
    return(s$v %*% as.matrix(diag(1 / s$d)) %*% t(s$u));
};

## Computes reconstruction error for some data, dimension range
## (dimRange), and number of runs.  Returns a data frame with columns
## "dims" for number of dimensions, "run" for the run number, and
## "err" for reconstruction error (as sum of squared error).
rcaReconstrError <- function(data, dimRange, runs) {
    foreach(dims=dimRange, .combine = "rbind") %:%
    foreach(run=1:runs, .combine='rbind') %dopar% {
        ## I'm not sure how kosher the below math is.
        ## projMtx <- randn(ncol(data), dims);
        ## reconstr <- projData %*% pinv(projMtx);
        projMtx <- RPGenerate(ncol(data), dims);
        projData <- as.matrix(data) %*% projMtx;
        reconstr <- projData %*% t(projMtx);
        return(data.frame(
            dims = dims,
            run = run,
            err = sum((reconstr - data)^2)
        ));
    };
}

## For input 'data' with P columns, returns a projection matrix of
## size (P, dims), computed by running random projections for the
## given number of runs.

## For input 'data' with P columns, runs random projections to 'dims'
## dimensions, for the given number of runs, returning a list with
## 'err' (a vector of the progressive lowest error across iterations)
## and 'mtx' (a projection matrix of size (P, dims) which produced the
## lowest error). Also, due to some idiotic bug in RPGenerate, don't
## use dims=1.
rcaBestProj <- function(data, dims, runs) {
    ## Get a list of lists (maybe there's a better way to do this, I
    ## don't really care), each one having 'err' (the reconstruction
    ## error), 'mtx' (the projection matrix producing that), and
    ## 'iter' (the iteration number).
    errs <- foreach(run=1:runs) %dopar% {
        projMtx <- RPGenerate(ncol(data), dims);
        projData <- as.matrix(data) %*% projMtx;
        reconstr <- projData %*% t(projMtx);
        return(list(iter = run,
                    err = sum((reconstr - data)^2),
                    mtx = projMtx));
    };
    ## Figure out the minimum error of all of these, and return the
    ## corresponding matrix:
    acc <- function(idxs, l) {
        idx <- idxs[length(idxs)];
        if (errs[[idx]]$err > l$err)
            return(c(idxs, l$iter))
        else
            return(c(idxs, idx));
    };
    iters <- Reduce(acc, errs, c(1));
    errHist <- sapply(iters, FUN=function(idx) errs[[idx]]$err);
    minErr <- errs[[iters[length(iters)]]];
    return(list(err = errHist, mtx = minErr$mtx));
}

###########################################################################
## Clustering re-projected points
###########################################################################

## For some data, corresponding output labels, a range of k values to
## use, and a number of iterations & runs for k-means, performs
## k-means clustering and computes the average classification error -
## if treating each cluster as representing the average label of its
## contents.
clusterLabelErrFrame <- function(data, labels, ks, iters, runs) {
    foreach(k=ks, .combine = "rbind") %do% {
        clusters <- kmeans(data, k, iters, runs);
        labelsAvg <- clusterPredictErr(clusters, labels);
        data.frame(k = k,
                   err = sum(labelsAvg$err) / nrow(labels));
    }
}

###########################################################################
## Neural nets
###########################################################################

## Given a neural network model from 'mlp' (of RSNNS), create a data
## frame which stacks the training & testing error.  Column 'idx' is
## iteration number, 'error' is the sum of squared error, and 'stage'
## is 'Train' or 'Test'.
stackError <- function(model)
{
    idxs <- seq(1, length(model$IterativeFitError));
    stacked <- rbind(
        data.frame(idx=idxs, error=model$IterativeFitError,  stage="Train"),
        data.frame(idx=idxs, error=model$IterativeTestError, stage="Test"));

    return(stacked);
}

###########################################################################
## K-Means outputs
###########################################################################

## For both datasets, across a range of k, get within-cluster
## sum-of-squared error and average silhouette value.
getKmeansClusters <- function() {
    fname <- "kmeansClusters.Rda";
    ks <- 2:300;
    iters <- 100;
    runs <- 50;
    runtime <- system.time(
        kmeansClusters <- foreach(k = ks, .combine='rbind') %dopar% {
            cat(k);
            cat("..");
            fc <- kmeans(faultsNorm,  k, iters, runs);
            lc <- kmeans(lettersNorm, k, iters, runs);
            fcSk <- silhouette(fc$cl, faultsDissim);
            lcSk <- silhouette(lc$cl, lettersDissim);
            fAvgLabels <- clusterPredictErr(fc, faultsLabels);
            lAvgLabels <- clusterPredictErr(lc, lettersLabels);
            return(data.frame(
                k = k,
                avgS = c(summary(fcSk)$avg.width,
                         summary(lcSk)$avg.width),
                withinSs = c(fc$tot.withinss / nrow(faultsNorm),
                             lc$tot.withinss / nrow(lettersNorm)),
                labelErr = c(sum(fAvgLabels$err) / nrow(faultsLabels),
                             sum(lAvgLabels$err) / nrow(lettersLabels)),
                test = c("Steel faults", "Letters"))
                )
        }
    )
    print(runtime);
    title <- sprintf("k-means, %d iters, %d runs",
                     iters, runs);
    xlab <- "k (number of clusters)";
    save(kmeansClusters, title, xlab, ylab, ks, iters, runs, runtime,
         file=fname);
};

## Produce a silhouette object to use elsewhere in plots:
getKmeansSilhouettes <- function() {
    fname <- "kmeansSilhouettes.Rda";
    kFaults <- 13;
    iters <- 100;
    runs <- 50;
    clusters <- kmeans(faultsNorm, kFaults, iters, runs);
    skFaults <- silhouette(clusters$cl, faultsDissim);
    titleFaults <- sprintf(
        "Steel faults, silhouette plot (k-means, k=%d, %d iters, %d runs)",
        kFaults, iters, runs);
    save(kFaults, skFaults, titleFaults, iters, runs, file=fname);
};

getKmeansConfusionMtx <- function() {
    fname <- "kmeansConfusionMtx.Rda";
    kFaults <- 200;
    kLetters <- 200;
    iters <- 100;
    runs <- 50;
    clustersF <- kmeans(faultsNorm, kLetters, iters, runs);
    clustersL <- kmeans(lettersNorm, kLetters, iters, runs);
    faultsConfusion <- clusterConfusionMatrix(clustersF, faultsLabels);
    lettersConfusion <- clusterConfusionMatrix(clustersL, lettersLabels);
    save(kLetters, kFaults, clustersF, clustersL, iters, runs,
         faultsConfusion, lettersConfusion, file=fname);
};

###########################################################################
## CFS or who knows what
###########################################################################
## Okay, that's atrocious.
## faultsFormula <- formula(paste(paste(labelNames, collapse=" + "), " ~ ."));

getCfsReconstrErr <- function() {
    fname <- "cfsReconstrError.Rda";
    faultsCfsTime <- system.time(
        faultsCfs <- cfs(Class ~ .,
                         cbind(faultsNorm,
                               data.frame(Class = faultsFactor)))
    );
    lettersCfsTime <- system.time(
        lettersCfs <- cfs(Letter ~ .,
                          cbind(lettersNorm,
                                data.frame(Letter = letters$Letter)))
    );

    faultsCfsErr <- data.frame(
        dims = c(1, ncol(faultsNorm)),
        err = sum((faultsNorm[-which(names(faultsNorm) %in% faultsCfs)])^2)
    );

    lettersCfsErr <- data.frame(
        dims = c(1, ncol(lettersNorm)),
        err = sum((lettersNorm[-which(names(lettersNorm) %in% lettersCfs)])^2)
    );

    cfsReconstrErr <- rbind(faultsCfsErr, lettersCfsErr);
    
    save(cfsReconstrErr, faultsCfs, faultsCfsTime, faultsCfsErr,
         lettersCfs, lettersCfsTime, lettersCfsErr, file=fname);
};

###########################################################################
## Reduced-dimensionality k-means
###########################################################################
runKmeansReducedDims <- function() {
    fname <- "kmeansReducedDims.Rda";
    iters <- 100;
    runs <- 50;

    rcaRuns <- 5000;
    
    ks <- c(200);

    origErr <-
        clusterLabelErrFrame(faultsNorm, faultsLabels, ks, iters, runs);
    ## Note the dimension range here; that is for the sake of plotting.
    kmeansOrigF <- data.frame(test = "Steel faults",
                              algo = "N/A",
                              dims = c(1, ncol(faultsNorm)),
                              err = origErr$err,
                              k = ks);

    rcaKmeansReducedF <-
        foreach(dims=2:26, .combine = "rbind") %do% {
            cat("RCA,F,");
            cat(dims);
            cat("..");
            rcaErr <- rcaBestProj(faultsNorm, dims, rcaRuns);
            mtx <- as.matrix(rcaErr$mtx);
            proj <- as.matrix(faultsNorm) %*% mtx;
            df <- clusterLabelErrFrame(proj, faultsLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Steel faults";
            df$algo <- "RCA";
            return(df);
        }
    
    ## TODO: Add timing information to this!
    faultsPca <- prcomp(faultsNorm);
    pcaKmeansReducedF <-
        foreach(dims=2:27, .combine = "rbind") %dopar% {
            cat("PCA,F,");
            cat(dims);
            cat("..");
            mtxR <- as.matrix(faultsPca$rotation[,1:dims]);
            proj <- as.matrix(faultsNorm) %*% mtxR;
            df <- clusterLabelErrFrame(proj, faultsLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Steel faults";
            df$algo <- "PCA";
            return(df);
        }

    if (FALSE) {
    kpcaKmeansReducedF <-
        foreach(dims=2:27, .combine = "rbind") %dopar% {
            cat("kpca");
            cat(dims);
            cat("..");
            kpc <- kpca(~., data=faultsNorm, kernel="rbfdot",
                        kpar=list(sigma=0.1), features=dims);
            proj <- rotated(kpc);
            df <- clusterLabelErrFrame(proj, faultsLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Steel faults";
            df$algo <- "k-PCA";
            return(df);
        }
    }
    
    icaKmeansReducedF <-
        foreach(dims=2:26, .combine = "rbind") %dopar% {
            cat("ICA,F,");
            cat(dims);
            cat("..");
            ica <- fastICA(faultsNorm, dims);
            ##A <- as.matrix(ica$A);
            ##S <- as.matrix(ica$S);
            ##proj <- S %*% A;
            df <- clusterLabelErrFrame(ica$S, faultsLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Steel faults";
            df$algo <- "ICA";
            return(df);
        }

    cfsKmeansReducedF <- local({
        faultsCfs <- cfs(Class ~ .,
                         cbind(faultsNorm,
                               data.frame(Class = faultsFactor)));
        proj <- faultsNorm[faultsCfs];
        df <- clusterLabelErrFrame(proj, faultsLabels, ks, iters, runs);
        ## Kludge alert (it's to plot properly):
        df1 <- df;
        df1$dims <- length(faultsCfs);
        df2 <- df;
        df2$dims <- ncol(faultsNorm);
        df <- rbind(df1, df2);
        df$test <- "Steel faults";
        df$algo <- "CFS";
        return(df);
    });

    ks <- c(200);
    origErr <-
        clusterLabelErrFrame(lettersNorm, lettersLabels, ks, iters, runs);
    ## Note the dimension range here; that is for the sake of plotting.
    kmeansOrigL <- data.frame(test = "Letters",
                              algo = "N/A",
                              dims = c(1, ncol(lettersNorm)),
                              err = origErr$err,
                              k = ks);

    rcaKmeansReducedL <-
        foreach(dims=2:16, .combine = "rbind") %do% {
            cat("RCA,L,");
            cat(dims);
            cat("..");
            rcaErr <- rcaBestProj(lettersNorm, dims, rcaRuns);
            mtx <- as.matrix(rcaErr$mtx);
            proj <- as.matrix(lettersNorm) %*% mtx;
            df <- clusterLabelErrFrame(proj, lettersLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Letters";
            df$algo <- "RCA";
            return(df);
        }
    
    lettersPca <- prcomp(lettersNorm);
    pcaKmeansReducedL <-
        foreach(dims=2:16, .combine = "rbind") %dopar% {
            cat("PCA,L,");
            cat(dims);
            cat("..");
            mtxR <- as.matrix(lettersPca$rotation[,1:dims]);
            proj <- as.matrix(lettersNorm) %*% mtxR;
            df <- clusterLabelErrFrame(proj, lettersLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Letters";
            df$algo <- "PCA";
            return(df);
        }

    ## This is so slow as to be unusable
    if (FALSE) {
    kpcaKmeansReducedL <-
        ## %do%, not %dopar%, because this is very memory-intensive:
        foreach(dims=2:16, .combine = "rbind") %do% {
            cat("kpca");
            cat(dims);
            cat("..");
            kpc <- kpca(~., data=lettersNorm, kernel="rbfdot",
                        kpar=list(sigma=0.1), features=dims);
            proj <- rotated(kpc);
            df <- clusterLabelErrFrame(proj, lettersLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Letters";
            df$algo <- "k-PCA";
            return(df);
        }
    }
    
    icaKmeansReducedL <-
        foreach(dims=2:16, .combine = "rbind") %dopar% {
            cat("ICA,L,");
            cat(dims);
            cat("..");
            ica <- fastICA(lettersNorm, dims);
            ##A <- as.matrix(ica$A);
            ##S <- as.matrix(ica$S);
            ##proj <- S %*% A;
            df <- clusterLabelErrFrame(ica$S, lettersLabels, ks, iters, runs);
            df$dims <- dims;
            df$test <- "Letters";
            df$algo <- "ICA";
            return(df);
        }

    cfsKmeansReducedL <- local({
        lettersCfs <- cfs(Letter ~ .,
                          cbind(lettersNorm,
                                data.frame(Letter = letters$Letter)));
        proj <- lettersNorm[lettersCfs];
        df <- clusterLabelErrFrame(proj, lettersLabels, ks, iters, runs);
        ## Kludge alert (it's to plot properly):
        df1 <- df;
        df1$dims <- length(lettersCfs);
        df2 <- df;
        df2$dims <- ncol(lettersNorm);
        df <- rbind(df1, df2);
        df$test <- "Letters";
        df$algo <- "CFS";
        return(df);
    });
    
    kmeansReducedDims <-
        rbind(kmeansOrigF, kmeansOrigL, pcaKmeansReducedF,
              icaKmeansReducedF, cfsKmeansReducedF, rcaKmeansReducedF,
              pcaKmeansReducedL, icaKmeansReducedL, rcaKmeansReducedL,
              cfsKmeansReducedL);
    ## TODO: Add 'k' value to this (though that's not present for EM)
    title <- "Cluster labels on dimension-reduced data";
    
    save(kmeansReducedDims, rcaRuns, ks, iters, runs, title,
         file=fname);
};

###########################################################################
## EM outputs
###########################################################################

## Build Mclust models for each dataset.  This will be very
## time-consuming.

getEmClusters <- function() {
    fname <- "emClusters.Rda";
    faultsMcTime <- system.time(
        faultsMc <- Mclust(faultsNorm, 2:500)
    );
    faultsBic <- data.frame(
        bic = faultsMc$BIC[,"EII"] / nrow(faultsNorm),
        numClusters = strtoi(names(faultsMc$BIC[,"EII"])),
        test = "Steel faults")
    faultsEmCc <- emConfusion(faultsMc, faultsLabels);
    ## The below is really, really slow.
    #lettersMcTime <- system.time(
    #    lettersMc <- Mclust(lettersNorm, seq(2, 300, by=5)
    #);
    ## These were run separately because it takes so damn long:
    load("lettersMc.Rda");
    load("lettersMc2.Rda");
    ## So, combine them together...
    lettersBicMtx <- as.data.frame(rbind(lettersMc$BIC, lettersMcPt2$BIC));
    ## and then find the max (which is a BIC) & argmax (which is a
    ## model, EII/VII/EEI/etc.) within each row:
    m <- apply(lettersBicMtx, 1, function(x) max(x, na.rm = TRUE));
    lettersBicMtx$bestModel <- factor(
        apply(lettersBicMtx, 1, which.max),
        levels = 1:ncol(lettersBicMtx),
        labels = colnames(lettersBicMtx));
    lettersBicMtx$bestBic <- m;
    lettersBic <- data.frame(
        bic = lettersBicMtx$bestBic / nrow(lettersNorm),
        numClusters = strtoi(rownames(lettersBicMtx)),
        test = "Letters");
    lettersEmCc <- emConfusion(lettersMc, lettersLabels);
    bic <- rbind(faultsBic, lettersBic);
    xlab <- "Number of clusters";
    ylab <- "Average BIC value";
    save(faultsMc, faultsMcTime, faultsEmCc, lettersMc, lettersEmCc,
         lettersMcTime, xlab, ylab, bic, file = fname);
};

## For both datasets, across a range of k, get within-cluster
## sum-of-squared error and average silhouette value.

getOptimalReducedClusters <- function() {
    fname <- "optimalReducedClusters.Rda";
    iters <- 100;
    runs <- 50;
    rcaRuns <- 5000;
    kFaults <- c(200);
    kLetters <- c(200);
    clRange <- seq(10, 160, by=5);
    faultsPcaDims <- 10;
    faultsIcaDims <- 10;
    faultsRcaDims <- 14;
    lettersPcaDims <- 10;
    lettersIcaDims <- 10;
    lettersRcaDims <- 13;

    print("Faults...");
    faultsClusters <- kmeans(faultsNorm, kFaults, iters, runs);
    faultsRefSse <- getKmeansSSE(faultsNorm, clRange, iters, runs);
    faultsRefSse$algo <- "N/A";
    
    print("Faults PCA...");
    faultsPcaTime <- system.time(faultsPca <- prcomp(faultsNorm));
    mtxR <- as.matrix(faultsPca$rotation[,1:faultsPcaDims]);
    faultsPcaClusters <- kmeans(as.matrix(faultsNorm) %*% mtxR,
                                kFaults, iters, runs);
    faultsPcaSse <- getKmeansSSE(as.matrix(faultsNorm) %*% mtxR,
                                 clRange, iters, runs);
    faultsPcaSse$algo <- "PCA";

    print("Faults ICA...");
    faultsIcaTime <- system.time(
        faultsIca <- fastICA(faultsNorm, faultsIcaDims));
    faultsIcaClusters <- kmeans(faultsIca$S, kFaults, iters, runs);
    faultsIcaSse <- getKmeansSSE(faultsIca$S, clRange, iters, runs);
    faultsIcaSse$algo <- "ICA";
    
    print("Faults RCA...");
    faultsRcaTime <- system.time(
        faultsRca <- rcaBestProj(faultsNorm, faultsRcaDims, rcaRuns));
    faultsRcaClusters <- kmeans(
        as.matrix(faultsNorm) %*% as.matrix(faultsRca$mtx),
        kFaults, iters, runs);
    faultsRcaSse <- getKmeansSSE(
        as.matrix(faultsNorm) %*% as.matrix(faultsRca$mtx), clRange, iters, runs);
    faultsRcaSse$algo <- "RCA";

    print("Faults CFS...");
    faultsCfsTime <- system.time(
        faultsCfs <- cfs(Class ~ .,
                         cbind(faultsNorm, data.frame(Class = faultsFactor))));
    faultsCfsClusters <-
        kmeans(faultsNorm[faultsCfs], kFaults, iters, runs);
    faultsCfsSse <- getKmeansSSE(faultsNorm[faultsCfs], clRange, iters, runs);
    faultsCfsSse$algo <- "CFS";

    inputs <- list(none = faultsNorm,
                   pca = as.matrix(faultsNorm) %*% mtxR,
                   ica = faultsIca$S,
                   rca = as.matrix(faultsNorm) %*% as.matrix(faultsRca$mtx),
                   cfs = faultsNorm[faultsCfs]);
    faultsEm <- foreach(input=inputs) %dopar% {
        cat("..");
        Mclust(input, clRange);
    };
    names(faultsEm) <- names(inputs);

    faultsSse <- rbind(faultsRefSse, faultsPcaSse, faultsIcaSse,
                       faultsRcaSse, faultsCfsSse);
    
    save(faultsClusters, faultsPcaClusters, faultsEm, faultsPcaTime,
         faultsIcaClusters, faultsIcaTime, faultsRcaClusters,
         faultsRcaTime, faultsCfsClusters, faultsCfsTime,
         faultsSse,
         file=fname);

    print("Letters...");
    lettersClusters <- kmeans(lettersNorm, kLetters, iters, runs);
    lettersRefSse <- getKmeansSSE(lettersNorm, clRange, iters, runs);
    lettersRefSse$algo <- "N/A";
    
    print("Letters PCA...");
    lettersPcaTime <- system.time(lettersPca <- prcomp(lettersNorm));
    mtxR <- as.matrix(lettersPca$rotation[,1:lettersPcaDims]);
    lettersPcaClusters <- kmeans(as.matrix(lettersNorm) %*% mtxR,
                                 kLetters, iters, runs);
    lettersPcaSse <- getKmeansSSE(as.matrix(lettersNorm) %*% mtxR,
                                 clRange, iters, runs);
    lettersPcaSse$algo <- "PCA";

    print("Letters ICA...");
    lettersIcaTime <- system.time(
        lettersIca <- fastICA(lettersNorm, lettersIcaDims));
    lettersIcaClusters <- kmeans(lettersIca$S, kLetters, iters, runs);
    lettersIcaSse <- getKmeansSSE(lettersIca$S, clRange, iters, runs);
    lettersIcaSse$algo <- "ICA";

    print("Letters RCA...");
    lettersRcaTime <- system.time(
        lettersRca <- rcaBestProj(lettersNorm, lettersRcaDims, rcaRuns));
    lettersRcaClusters <- kmeans(
        as.matrix(lettersNorm) %*% as.matrix(lettersRca$mtx),
        kLetters, iters, runs);
    lettersRcaSse <- getKmeansSSE(
        as.matrix(lettersNorm) %*% as.matrix(lettersRca$mtx), clRange, iters, runs);
    lettersRcaSse$algo <- "RCA";

    print("Letters CFS...");
    lettersCfsTime <- system.time(
        lettersCfs <- cfs(Class ~ .,
                          cbind(lettersNorm,
                                data.frame(Class = letters$Letter))));
    lettersCfsClusters <-
        kmeans(lettersNorm[lettersCfs], kLetters, iters, runs);
    lettersCfsSse <- getKmeansSSE(lettersNorm[lettersCfs], clRange, iters, runs);
    lettersCfsSse$algo <- "CFS";

    lettersSse <- rbind(lettersRefSse, lettersPcaSse, lettersIcaSse,
                       lettersRcaSse, lettersCfsSse);
    save(faultsSse, lettersSse, file="reducedSse.Rda");
    
    inputs <- list(pca = as.matrix(lettersNorm) %*% mtxR,
                   ica = lettersIca$S,
                   rca = as.matrix(lettersNorm) %*% as.matrix(lettersRca$mtx),
                   cfs = lettersNorm[lettersCfs]);
    lettersEm <- foreach(input=inputs) %dopar% {
        Mclust(input, clRange);
    };
    names(lettersEm) <- names(inputs);
    
    save(faultsClusters, faultsPcaClusters, faultsEm, faultsPcaTime,
         faultsIcaClusters, faultsIcaTime, faultsRcaClusters,
         faultsRcaTime, faultsCfsClusters, faultsCfsTime,  faultsSse,
         lettersClusters, lettersPcaClusters, lettersEm,
         lettersPcaTime, lettersIcaClusters, lettersIcaTime,
         lettersRcaClusters, lettersRcaTime, lettersCfsClusters,
         lettersCfsTime, lettersSse, file=fname);
};

getClusterSpectrum <- function() {
    fname <- "clusterSpectrum.Rda";
    load("optimalReducedClusters.Rda");

    inputs <- list(list("N/A", faultsClusters),
                   list("PCA", faultsPcaClusters),
                   list("ICA", faultsIcaClusters),
                   list("RCA", faultsRcaClusters),
                   list("CFS", faultsCfsClusters));
    faultsSpectrum <- foreach (input = inputs, .combine = "rbind") %do% {
        algo <- input[[1]];
        cl <- input[[2]];
        data.frame(
            algo  = algo,
            idx   = 1:length(cl$size),
            sizes = sort(cl$size, decreasing = TRUE) / sum(cl$size));
    };

    inputs <- list(list("N/A", lettersClusters),
                   list("PCA", lettersPcaClusters),
                   list("ICA", lettersIcaClusters),
                   list("RCA", lettersRcaClusters),
                   list("CFS", lettersCfsClusters));
    lettersSpectrum <- foreach (input = inputs, .combine = "rbind") %do% {
        algo <- input[[1]];
        cl <- input[[2]];
        data.frame(
            algo  = algo,
            idx   = 1:length(cl$size),
            sizes = sort(cl$size, decreasing = TRUE) / sum(cl$size));
    };
    
    save(faultsSpectrum, lettersSpectrum, file = fname);
};

getBicCurves <- function() {
    fname <- "bicCurves.Rda";
    load("optimalReducedClusters.Rda");

    faultsBic <- emGetBic(faultsEm$none);
    faultsBic$algo <- "N/A";
    
    faultsPcaBic <- emGetBic(faultsEm$pca);
    faultsPcaBic$algo <- "PCA";
    
    faultsIcaBic <- emGetBic(faultsEm$ica);
    faultsIcaBic$algo <- "ICA";
    
    faultsRcaBic <- emGetBic(faultsEm$rca);
    faultsRcaBic$algo <- "RCA";
    
    faultsCfsBic <- emGetBic(faultsEm$cfs);
    faultsCfsBic$algo <- "CFS";

    ## TODO: Add letters to this?
    faultsReducedBic <- rbind(faultsBic, faultsPcaBic, faultsIcaBic,
                              faultsRcaBic, faultsCfsBic);
    
    save(faultsReducedBic, file = fname);
};

###########################################################################
## PCA outputs
###########################################################################

faultsPca <- prcomp(faultsNorm);
lettersPca <- prcomp(lettersNorm);

local({
    faultsPcaDf <- data.frame(class = faultsFactor, pca = faultsPca$x);
    lettersPcaDf <- data.frame(class = letters$Letter, pca = lettersPca$x);
    save(faultsPca, lettersPca, faultsPcaDf, lettersPcaDf, file="pca.Rda");
});

## This might need redone later to compare other reduction techniques
getPcaReconstrErr <- function() {
    fname <- "pcaReconstrError.Rda";
    faultsMtx <- faultsPca$rotation;
    lettersMtx <- lettersPca$rotation;
    faultsPcaErr <- data.frame(
        dims = 1:nrow(faultsMtx),
        err = reconstrError(faultsMtx, faultsNorm) / nrow(faultsNorm),
        test = "Steel faults");
    lettersPcaErr <- data.frame(
        dims = 1:nrow(lettersPca$rotation),
        err = reconstrError(lettersMtx, lettersNorm) / nrow(lettersNorm),
        test = "Letters");
    pcaReconstrErr <- rbind(faultsPcaErr, lettersPcaErr);
    title <- "PCA reconstruction error";
    xlab <- "Dimensions";
    ylab <- "Average reconstruction error";
    save(pcaReconstrErr, title, xlab, ylab, file=fname);
};

## Now, this might give some notion of the contribution of each
## original component up to the indication principal:
contrib <- apply(faultsPca$rotation^2, 1, cumsum);
## (That is, row 'i' stands for the i'th principal component and
## column 'j' for the j'th feature, and the value at (i,j) is how much
## feature 'j' has contributed to the first 'i' principals.)
rownames(contrib) <- NULL;
contribStacked <- stack(data.frame(t(contrib)));
## We need to know which feature produced each row in this 'stacked'
## data frame, so do this with R's recycling behavior:
contribStacked$feature <- colnames(contrib);
## Column 'ind' is not numerical, and it is sorting wrong.  It's of
## the format "P1", "P2", and so on - so remove the first character,
## and evaluate to an integer.
contribStacked$ind <- strtoi(substring(contribStacked$ind, 2));
## This is really kludgy but I don't know how to get around it.

##ggplot(data=contribStacked,
##       aes(x=ind, y=values, group=feature)) +
##    geom_line(aes(colour=feature)) +
##    xlab("Principal component");

###########################################################################
## ICA outputs
###########################################################################

local({
    faultsIca <- fastICA(faultsNorm, 2);
    lettersIca <- fastICA(lettersNorm, 2);
    faultIcaDf <- data.frame(class = faultsFactor, ica = faultsIca$S);
    lettersIcaDf <- data.frame(class = letters$Letter, ica = lettersIca$S);
    save(faultsIca, lettersIca, faultIcaDf, lettersIcaDf, file="ica.Rda");
});

getIcaReconstrErr <- function() {
    fname <- "icaReconstrError.Rda";
    faultsIcaTime <- system.time(
        faultsIcaErr <- icaReconstrError(faultsNorm, 1:26)
    );
    faultsIcaErr$err = faultsIcaErr$err / nrow(faultsNorm);
    faultsIcaErr$test = "Steel faults";
    lettersIcaTime <- system.time(
        lettersIcaErr <- icaReconstrError(lettersNorm, 1:16)
    );
    lettersIcaErr$err = lettersIcaErr$err / nrow(lettersNorm);
    lettersIcaErr$test = "Letters";
    icaReconstrErr <- rbind(faultsIcaErr, lettersIcaErr);
    title <- "ICA reconstruction error";
    xlab <- "Dimensions";
    ylab <- "Average reconstruction error";
    save(icaReconstrErr, faultsIcaTime, lettersIcaTime, xlab, ylab, title,
         file = fname);
};

###########################################################################
## RCA outputs
###########################################################################

getRcaDf <- function() {
    faultsRca <- rcaBestProj(faultsNorm, 2, 5000);
    lettersRca <- rcaBestProj(lettersNorm, 2, 5000);
    faultsRcaDf <- data.frame(class = faultsFactor,
                              rca = as.matrix(faultsNorm) %*% faultsRca$mtx);
    lettersRcaDf <- data.frame(class = letters$Letter,
                               rca = as.matrix(lettersNorm) %*% lettersRca$mtx);
    save(faultsRca, lettersRca, faultsRcaDf, lettersRcaDf, file="rca.Rda");
};

##decim <- lettersRcaDf[seq(1,nrow(lettersRcaDf),by=30),];
##ggplot(decim, aes(label = class, rca.1, rca.2, colour = class)) +
##    geom_text() +
##    ggtitle("Letters: First two PCA components")

    #geom_point(alpha = 0.5, size = 2) +
    ##scale_shape_manual(values=as.numeric(lettersPcaDf$class))

getRcaErr <- function() {
    fname <- "rcaErrorCurve.Rda";

    runs <- 20000;
    faultsRcaTime <- system.time(
        faultsRcaErr <-
            foreach (dims = 2:26, .combine = rbind) %do% {
                cat(dims);
                cat('..');
                rca <- rcaBestProj(faultsNorm, dims, runs);
                return(data.frame(iter = 1:length(rca$err),
                                  err  = rca$err / nrow(faultsNorm),
                                  dims = dims,
                                  test = "Steel faults"));
            }
    );

    lettersRcaTime <- system.time(
        lettersRcaErr <-
            foreach (dims = 2:16, .combine = rbind) %do% {
                cat(dims);
                cat('..');
                rca <- rcaBestProj(lettersNorm, dims, runs);
                return(data.frame(iter = 1:length(rca$err),
                                  err  = rca$err / nrow(lettersNorm),
                                  dims = dims,
                                  test = "Letters"));
            }
    );

    rcaErr <- rbind(faultsRcaErr, lettersRcaErr);
    save(rcaErr, lettersRcaTime, faultsRcaTime, file=fname);
};

getRcaReconstrErr <- function() {
    fname <- "rcaReconstrError.Rda";
    runs <- 5000;
    faultsRcaTime <- system.time(
        faultsRcaErr <- rcaReconstrError(faultsNorm, 2:26, runs)
    );
    faultsRcaErr$err = faultsRcaErr$err / nrow(faultsNorm);
    faultsRcaErr$test = "Steel faults";
    lettersRcaTime <- system.time(
        lettersRcaErr <- rcaReconstrError(lettersNorm, 2:16, runs)
    );
    lettersRcaErr$err = lettersRcaErr$err / nrow(lettersNorm);
    lettersRcaErr$test = "Letters";
    rcaReconstrErr <- rbind(faultsRcaErr, lettersRcaErr);
    title <- "RCA reconstruction error";
    xlab <- "Dimensions";
    ylab <- "Average reconstruction error";
    save(rcaReconstrErr, faultsRcaTime, lettersRcaTime, runs, xlab, ylab,
         title, file = fname);
};

###########################################################################
## MDS (Multidimensional Scaling) or whatever
###########################################################################
failedTests <- function() {
    faultsMds <- cmdscale(faultsDissim, 2, eig = TRUE);
    ## lettersMds <- cmdscale(lettersDissim, 2, eig = TRUE);
    faultsMdsDf <- data.frame(
        x = faultsMds$points[, 1],
        y = faultsMds$points[, 2],
        class = faultsFactor
    );

    ggplot(faultsMdsDf) +
        geom_point(aes(x, y, colour = class), size = 2);

    ## Faults data is lower-rank for some reason, so remove one column:
    ## rmCol <- c(-1, -2, -3, -4, -5, -6, -7);
    ## faultsLda <- lda(faultsFormula, cbind(faultsNorm[,rmCol], faultsLabels));

    lettersLda <- lda(Letter ~ .,
                      cbind(lettersNorm,
                            data.frame(Letter = letters$Letter)));

    plda <- predict(object = lettersLda, newdata = lettersNorm);
    dataset <- data.frame(class = letters$Letter,
                          lda = plda$x);

    p1 <- ggplot(dataset) +
        geom_point(aes(lda.LD1, lda.LD2, colour = class), size = 2.5)
    print(p1)

    kpc <- kpca(~.,
                data=faultsNorm,
                kernel="rbfdot",
                kpar=list(sigma=0.1),
                features=2);

    kpc10 <- kpca(~.,
                  data=faultsNorm,
                  kernel="rbfdot",
                  kpar=list(sigma=0.1),
                  features=10);

    faultsKpcaDf <- data.frame(
        x = rotated(kpc)[, 1],
        y = rotated(kpc)[, 2],
        class = faultsFactor
    );

    ggplot(faultsKpcaDf) +
        geom_point(aes(x, y, colour = class), size = 2);
};
 
###########################################################################
## Neural net outputs
###########################################################################

## Train a 'reference' net on the original data:
faultsNn <- mlp(faultsTrain[inputNames], faultsTrain[labelNames],
                size = 30, learnFuncParams = c(0.6, 0.0),
                maxit = 100, inputsTest = faultsTest[inputNames],
                targetsTest = faultsTest[labelNames]);

## So far the below just tests PCA:
getNnetLearning <- function(){
    fname <- "nnetLearning.Rda"
    neuralNetTime <- system.time(
        neuralNetErr <-
            foreach (dims = c(5, 10, 15), .combine = rbind) %dopar% {
                cat(dims);
                cat('..');
                faultsPca <- prcomp(faultsNorm);
                mtxR <- as.matrix(faultsPca$rotation[,1:dims]);
                projTrain <- as.matrix(faultsTrain[inputNames]) %*% mtxR;
                projTest <- as.matrix(faultsTest[inputNames]) %*% mtxR;
                model <- mlp(projTrain,
                             faultsTrain[labelNames],
                             size = 30,
                             learnFuncParams = c(0.6, 0.0),
                             maxit = 100,
                             inputsTest = projTest,
                             targetsTest = faultsTest[labelNames]);
                err <- stackError(model);
                err$dims <- dims;
                return(err);
        }
    );
    neuralNetErr$dims <- sprintf("%d", neuralNetErr$dims);
    err <- stackError(faultsNn);
    err$dims <- "N/A";
    neuralNetErr <- rbind(neuralNetErr, err);
    
    save(neuralNetErr, neuralNetTime, file=fname);
};

## This is specific to the steel faults data set:
nnTrain <- function(train, test, maxit) {
    runs <- 20;
    errs <- foreach (run = 1:runs, .combine = rbind) %dopar% {
        cat(run);
        cat(".");
        model <- mlp(train,
                     faultsTrain[labelNames],
                     size = 30,
                     learnFuncParams = c(0.6, 0.0),
                     maxit = maxit,
                     inputsTest = test,
                     targetsTest = faultsTest[labelNames]);
        cat(".");
        ## Apply to training & test dataset to get errors:
        trainOutput <- predict(model, train);
        trainIdxs <- apply(faultsTrain[labelNames], 1, which.max);
        idxs <- apply(trainOutput, 1, which.max);
        trainCorrect <- sum(idxs == trainIdxs);
        trainErr <- 1 - trainCorrect / length(trainIdxs);
        testOutput  <- predict(model, test);
        testIdxs <- apply(faultsTest[labelNames], 1, which.max);
        idxs <- apply(testOutput, 1, which.max);
        testCorrect <- sum(idxs == testIdxs);
        testErr <- 1 - testCorrect / length(testIdxs);
        cat(".");
        return(c(trainErr, testErr));
    };
    means <- apply(errs, 1, mean);
    stdevs <- apply(errs, 1, sd);
    return(data.frame(trainErr = means[1],
                      testErr = means[2],
                      trainErrSd = stdevs[1],
                      testErrSd = stdevs[2]));
};

getNnetError <- function() {
    fname <- "nnetError.Rda"

    ## Train another 'reference' net on original data, but with the more
    ## optimal number of iterations:
    d1 <- ncol(faultsNorm);
    r <- nnTrain(faultsTrain[inputNames], faultsTest[inputNames], 15);
    refErr <- data.frame(
        dims = c(1, 1, d1, d1),
        algo = "N/A",
        error = c(r$trainErr, r$testErr, r$trainErr, r$testErr),
        errorSd = c(r$trainErrSd, r$testErrSd, r$trainErrSd, r$testErrSd),
        stage = c("Train", "Test", "Train", "Test"));

    rcaNnErr <-
        foreach (dims = 2:26, .combine = rbind) %do% {
            cat(dims);
            cat('..');
            rcaErr <- rcaBestProj(faultsNorm, dims, 10000);
            mtx <- as.matrix(rcaErr$mtx);
            projTrain <- as.matrix(faultsTrain[inputNames]) %*% mtx;
            projTest <- as.matrix(faultsTest[inputNames]) %*% mtx;
            err <- nnTrain(projTrain, projTest, 15);
            return(data.frame(dims = dims,
                              algo = "RCA",
                              error = c(err$trainErr, err$testErr),
                              errorSd = c(err$trainErrSd, err$testErrSd),
                              stage = c("Train", "Test")));
        };
    
    pcaNnErr <-
        foreach (dims = 1:27, .combine = rbind) %do% {
            cat(dims);
            cat('..');
            faultsPca <- prcomp(faultsNorm);
            mtxR <- as.matrix(faultsPca$rotation[,1:dims]);
            projTrain <- as.matrix(faultsTrain[inputNames]) %*% mtxR;
            projTest <- as.matrix(faultsTest[inputNames]) %*% mtxR;
            err <- nnTrain(projTrain, projTest, 15);
            return(data.frame(dims = dims,
                              algo = "PCA",
                              error = c(err$trainErr, err$testErr),
                              errorSd = c(err$trainErrSd, err$testErrSd),
                              stage = c("Train", "Test")));
        };

    icaNnErr <-
        foreach (dims = 1:26, .combine = rbind) %do% {
            cat(dims);
            cat('..');
            ica <- fastICA(faultsNorm, dims);
            ##A <- as.matrix(ica$A);
            ##S <- as.matrix(ica$S);
            proj <- ica$K %*% ica$W;
            projTrain <- as.matrix(faultsTrain[inputNames]) %*% proj;
            projTest <- as.matrix(faultsTest[inputNames]) %*% proj;
            err <- nnTrain(projTrain, projTest, 15);
            return(data.frame(dims = dims,
                              algo = "ICA",
                              error = c(err$trainErr, err$testErr),
                              errorSd = c(err$trainErrSd, err$testErrSd),
                              stage = c("Train", "Test")));
        };

    cfsNnErr <- local({
        faultsCfs <- cfs(Class ~ .,
                         cbind(faultsNorm,
                               data.frame(Class = faultsFactor)));
        projTrain <- faultsTrain[faultsCfs];
        projTest <- faultsTest[faultsCfs];
        err <- nnTrain(projTrain, projTest, 15);
        ## Same kludge (to plot properly):
        d0 <- length(faultsCfs);
        d1 <- ncol(faultsNorm);
        return(data.frame(
            dims = c(d0, d0, d1, d1),
            algo = "CFS",
            error = c(err$trainErr, err$testErr, err$trainErr, err$testErr),
            errorSd = c(err$trainErrSd, err$testErrSd),
            stage = c("Train", "Test", "Train", "Test")));
    });

    neuralNetErr <- rbind(rcaNnErr, pcaNnErr, icaNnErr, cfsNnErr, refErr);
    
    save(neuralNetErr, refErr, file=fname);
};

###########################################################################
## Neural nets using clusters as input
###########################################################################
getNnetClusterLearningCurve <- function() {
    fname <- "nnetClusterLearningCurve.Rda";
    load("optimalReducedClusters.Rda");

    ## Get indices of training data set:
    trainIdxs <- tt$idx;
    ## For consistency, we keep this to how the other datasets were
    ## generated, since we end up with exactly the same number of instances.

    runMlp <- function(train, test) {
        runs <- 20;
        err <- foreach(run = 1:runs, .combine="rbind") %dopar% {
            cat(run);
            cat("..");
            model <- mlp(train,
                         faultsTrain[labelNames],
                         size = 30,
                         learnFuncParams = c(0.6, 0.0),
                         maxit = 200,
                         inputsTest = test,
                         targetsTest = faultsTest[labelNames]);
            return(stackError(model));
        };
        errMean <- aggregate(error ~ stage + idx, err, mean);
        errSd <- aggregate(error ~ stage + idx, err, sd);
        return(data.frame(idx     = errMean$idx,
                          errMean = errMean$err,
                          stage   = errMean$stage,
                          errSd   = errSd$err));
    };

    ## Is there some better way to do this?
    inputs <- list(
        list(none=faultsClusters, type="k"),
        list(none=faultsEm$none, type="em")
        ## So, that was all totally unnecessary:
        ##list(PCA=faultsPcaClusters, type="k"),
        ##list(ICA=faultsIcaClusters, type="k"),
        ##list(RCA=faultsRcaClusters, type="k"),
        ##list(CFS=faultsCfsClusters, type="k"),
        ##list(PCA=faultsEm$pca,      type="em"),
        ##list(RCA=faultsEm$rca,      type="em"),
        ##list(ICA=faultsEm$ica,      type="em"),
        ##list(CFS=faultsEm$cfs,      type="em")
        );
    
    nnetClusterLearningCurve <- foreach(input=inputs, .combine="rbind") %do% {
        algo <- names(input)[1];
        cat(names(input)[1]);
        cat(".");
        if (input$type == "k") {
            ## Turn class labels into a binary vector:
            cl <- decodeClassLabels(input[[1]]$cluster);
        } else {
            ## EM's probabilities are already in the form we need:
            cl <- input[[1]]$z;
        }
        cat(".");
        ## Split into training & test sets:
        train <- cl[trainIdxs,];
        test <- cl[-trainIdxs,];
        ## Test with just the cluster labels as inputs:
        err1 <- runMlp(train, test);
        err1$algo <- algo;
        err1$inputs <- "Clusters";
        err1$type <- input$type;
        ## And with the cluster labels *and* normal inputs:
        err2 <- runMlp(cbind(train, faultsTrain[inputNames]),
                       cbind(test,  faultsTest[inputNames]));
        err2$algo <- algo;
        err2$inputs <- "Both";
        err2$type <- input$type;
        return(rbind(err1, err2));
    };

    ## Get 'reference' error too from original data
    refErr <- runMlp(faultsTrain[inputNames], faultsTest[inputNames]);
    refErr$algo <- "N/A";
    refErr$inputs <- "N/A";
    refErr$type <- "N/A";
    nnetClusterLearningCurve <- rbind(nnetClusterLearningCurve, refErr);
    
    save(nnetClusterLearningCurve, file=fname);
};

getNnetClusterErrs <- function() {
    fname <- "nnetClusterErrs.Rda";
    load("optimalReducedClusters.Rda");

    ## Get indices of training data set:
    trainIdxs <- tt$idx;
    ## For consistency, we keep this to how the other datasets were
    ## generated, since we end up with exactly the same number of instances.

    ref <- nnTrain(faultsTrain[inputNames], faultsTest[inputNames], 30);
    ref$algo <- "N/A";
    ref$inputs <- "N/A";
    ref$type <- "N/A";
    
    ## Is there some better way to do this?
    inputs <- list(
        list(none=faultsClusters, type="k"),
        list(none=faultsEm$none, type="em")
        ## So, that was all totally unnecessary:
        ##list(PCA=faultsPcaClusters, type="k"),
        ##list(ICA=faultsIcaClusters, type="k"),
        ##list(RCA=faultsRcaClusters, type="k"),
        ##list(CFS=faultsCfsClusters, type="k"),
        ##list(PCA=faultsEm$pca,      type="em"),
        ##list(RCA=faultsEm$rca,      type="em"),
        ##list(ICA=faultsEm$ica,      type="em"),
        ##list(CFS=faultsEm$cfs,      type="em")
        );

    ## Run neural networks using just the clusters as inputs
    nnetClusterErrs <- foreach(input=inputs, .combine="rbind") %do% {
        algo <- names(input)[1];
        cat(algo);
        cat(".");
        if (input$type == "k") {
            ## Turn class labels into a binary vector:
            cl <- decodeClassLabels(input[[1]]$cluster);
        } else {
            ## EM's probabilities are already in the form we need:
            cl <- input[[1]]$z;
        }
        cat(".");
        ## Split into training & test sets:
        train <- cl[trainIdxs,];
        test <- cl[-trainIdxs,];
        ## Test using just the labels as input:
        err1 <- nnTrain(train, test, 50);
        err1$inputs <- "Clusters";
        err1$algo <- algo;
        err1$type <- input$type;
        ## Test using the cluster labels *and* normal inputs:
        err2 <- nnTrain(cbind(train, faultsTrain[inputNames]),
                        cbind(test,  faultsTest[inputNames]),
                        50);
        err2$inputs <- "Both";
        err2$algo <- algo;
        err2$type <- input$type;
        cat(".");
        return(rbind(err1, err2));
    };
    nnetClusterErrs <- rbind(nnetClusterErrs, ref);
    
    save(nnetClusterErrs, file=fname);
};

## runKmeansReducedDims();
## getNnetError();
## getNnetClusterLearningCurve();
## getNnetClusterErrs();
getOptimalReducedClusters();
