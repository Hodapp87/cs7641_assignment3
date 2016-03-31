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
## Data Loading & Other Boilerplate
###########################################################################

## Split ratio 'f' for training data, and the rest (1-f) for testing.
## Returns a list with items "train" and "test" containing rows taken,
## in random order, from "frame".
splitTrainingTest <- function(frame, f) {
    trainIdx <- sample(nrow(frame), size=f*nrow(frame));
    train <- frame[trainIdx,];
    test <- frame[-trainIdx,];
    return(list(train = train, test = test));
}

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
faultFactor <- factor(apply(faultsLabels, 1, function(x) which(x == 1)),
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
    return(conf);
}

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

## Works, but plots aren't particularly useful:
## dissMtx <- dist(clusters$centers, upper = TRUE, diag = TRUE);
## clustersHist <- distHistogram(clusters, faultsNorm, clusters$cluster);
## ggplot(data=clustersHist,
##        aes(x=bins, y=hist, group=factor(class))) +
##     geom_line(aes(colour=factor(class))) +
##     xlab("Distance to cluster center") +
##     ylab("Cumulative probability") +
##     ggtitle("Distance distribution in each cluster")

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
    return(confusionMatrix(correct, predicted));
}

###########################################################################
## PCA
###########################################################################

faultsPca <- prcomp(faultsNorm);
lettersPca <- prcomp(lettersNorm);

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

faultsIca <- foreach(dims=2:26) %dopar% fastICA(faultsNorm, dims);
lettersIca <- foreach(dims=2:16) %dopar% fastICA(lettersNorm, dims);
## But how do I get dimensions into here?

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
## lowest error).
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
    kLetters <- 200;
    iters <- 100;
    runs <- 50;
    clusters <- kmeans(lettersNorm, kLetters, iters, runs);
    lettersConfusion <- clusterConfusionMatrix(clusters, lettersLabels);
    save(kLetters, iters, runs, lettersConfusion, file=fname);
};

###########################################################################
## CFS or who knows what
###########################################################################
## Okay, that's atrocious.
## faultsFormula <- formula(paste(paste(labelNames, collapse=" + "), " ~ ."));

local({
    fname <- "cfs.Rda";
    faultsCfs <- cfs(Class ~ .,
                     cbind(faultsNorm,
                           data.frame(Class = faultFactor)));
    lettersCfs <- cfs(Letter ~ .,
                      cbind(lettersNorm,
                            data.frame(Letter = letters$Letter)));

    save(faultsCfs, lettersCfs, file=fname);
});

###########################################################################
## Reduced-dimensionality k-means
###########################################################################
runKmeansReducedDims <- function() {
    fname <- "kmeansReducedDims.Rda";
    iters <- 100;
    runs <- 50;

    ks <- c(20, 80, 250);

    kmeansOrigF <-
        clusterLabelErrFrame(faultsNorm, faultsLabels, ks, iters, runs);
    kmeansOrigF$test <- "Steel faults";
    kmeansOrigF$algo <- "None";

    ##origClusters <- kmeans(faultsNorm, kFaults, iters, runs);
    
    ## TODO: Add timing information to this!
    faultsPca <- prcomp(faultsNorm);
    pcaKmeansReducedF <-
        foreach(dims=2:27, .combine = "rbind") %dopar% {
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
                               data.frame(Class = faultFactor)));
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
    
    kmeansOrigL <-
        clusterLabelErrFrame(lettersNorm, lettersLabels, ks, iters, runs);
    kmeansOrigL$test <- "Letters";
    kmeansOrigL$algo <- "None";

    lettersPca <- prcomp(lettersNorm);
    pcaKmeansReducedL <-
        foreach(dims=2:16, .combine = "rbind") %dopar% {
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
        rbind(pcaKmeansReducedF, icaKmeansReducedF, cfsKmeansReducedF,
              pcaKmeansReducedL, icaKmeansReducedL, cfsKmeansReducedL);
    ## TODO: Add 'k' value to this (though that's not present for EM)
    title <- "Cluster labels on dimension-reduced data";
    
    save(kmeansReducedDims, kmeansOrigF, kmeansOrigL, title, file=fname);
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
    #    lettersMc <- Mclust(lettersNorm, 2:100)
    #);
    load("lettersMc.Rda");
    lettersBic <- data.frame(
        bic = lettersMc$BIC[,"EII"] / nrow(lettersNorm),
        numClusters = strtoi(names(lettersMc$BIC[,"EII"])),
        test = "Letters")
    lettersEmCc <- emConfusion(lettersMc, lettersLabels);
    bic <- rbind(faultsBic, lettersBic);
    xlab <- "Number of clusters";
    ylab <- "Average BIC value";
    save(faultsMc, faultsMcTime, faultsEmCc, lettersMc, lettersEmCc,
         lettersMcTime, xlab, ylab, bic, file = fname);
};

###########################################################################
## PCA outputs
###########################################################################

local({
    faultPcaDf <- data.frame(class = faultFactor, pca = faultsPca$x);
    lettersPcaDf <- data.frame(class = letters$Letter, pca = lettersPca$x);
    save(faultsPca, lettersPca, faultPcaDf, lettersPcaDf, file="pca.Rda");
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

ggplot(data=contribStacked,
       aes(x=ind, y=values, group=feature)) +
    geom_line(aes(colour=feature)) +
    xlab("Principal component");

###########################################################################
## ICA outputs
###########################################################################

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
        class = faultFactor
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
        class = faultFactor
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
local({
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
});

nnTrain <- function(train, test) {
    model <- mlp(train,
                 faultsTrain[labelNames],
                 size = 30,
                 learnFuncParams = c(0.6, 0.0),
                 maxit = 15,
                 inputsTest = test,
                 targetsTest = faultsTest[labelNames]);
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
    return(data.frame(trainErr, testErr));
};

local({
    fname <- "nnetError.Rda"

    ## Train another 'reference' net on original data, but with the more
    ## optimal number of iterations:
    d1 <- ncol(faultsNorm);
    r <- nnTrain(faultsTrain[inputNames], faultsTest[inputNames]);
    refErr <- data.frame(
        dims = c(1, 1, d1, d1),
        algo = "N/A",
        error = c(r$trainErr, r$testErr, r$trainErr, r$testErr),
        stage = c("Train", "Test", "Train", "Test"));
        
    pcaNnErr <-
        foreach (dims = 1:27, .combine = rbind) %dopar% {
            cat(dims);
            cat('..');
            faultsPca <- prcomp(faultsNorm);
            mtxR <- as.matrix(faultsPca$rotation[,1:dims]);
            projTrain <- as.matrix(faultsTrain[inputNames]) %*% mtxR;
            projTest <- as.matrix(faultsTest[inputNames]) %*% mtxR;
            err <- nnTrain(projTrain, projTest);
            return(data.frame(dims = dims,
                              algo = "PCA",
                              error = c(err$trainErr, err$testErr),
                              stage = c("Train", "Test")));
        };

    icaNnErr <-
        foreach (dims = 1:26, .combine = rbind) %dopar% {
            cat(dims);
            cat('..');
            ica <- fastICA(faultsNorm, dims);
            ##A <- as.matrix(ica$A);
            ##S <- as.matrix(ica$S);
            proj <- ica$K %*% ica$W;
            projTrain <- as.matrix(faultsTrain[inputNames]) %*% proj;
            projTest <- as.matrix(faultsTest[inputNames]) %*% proj;
            err <- nnTrain(projTrain, projTest);
            return(data.frame(dims = dims,
                              algo = "ICA",
                              error = c(err$trainErr, err$testErr),
                              stage = c("Train", "Test")));
        };

    cfsNnErr <- local({
        faultsCfs <- cfs(Class ~ .,
                         cbind(faultsNorm,
                               data.frame(Class = faultFactor)));
        projTrain <- faultsTrain[faultsCfs];
        projTest <- faultsTest[faultsCfs];
        err <- nnTrain(projTrain, projTest);
        ## Same kludge (to plot properly):
        d0 <- length(faultsCfs);
        d1 <- ncol(faultsNorm);
        return(data.frame(
            dims = c(d0, d0, d1, d1),
            algo = "CFS",
            error = c(err$trainErr, err$testErr, err$trainErr, err$testErr),
            stage = c("Train", "Test", "Train", "Test")));
    });

    neuralNetErr <- rbind(pcaNnErr, icaNnErr, cfsNnErr, refErr);
    
    save(neuralNetErr, refErr, file=fname);
});
