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

source("multiplot.R");

###########################################################################
## Data Loading & Other Boilerplate
###########################################################################

print("Loading & converting data...");
# pdf("assignment1_plots.pdf");

## Load data for "Steel Plates Faults" & apply proper header names:
faults <- read.table("Faults.NNA", sep="\t", header=FALSE);
colnames(faults) <- readLines("Faults27x7_var");
## If these are missing, use:
## https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults
## https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA:
## https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults27x7_var

## Dependent variables are several exclusive binary columns:
depCol <- c("Pastry", "Z_Scratch", "K_Scatch", "Stains",
            "Dirtiness", "Bumps", "Other_Faults");
## Turn these into a factor:
## resp <- factor(apply(faults[depCol], 1, function(x) which(x == 1)),
##                labels = depCol);

## Set that factor to 'Fault' and remove the variables that created it:
## faults$Fault <- resp;
## faults[depCol] <- list(NULL);

## Also standardize data to mean 0, variance 1, separating labels:
faultsNorm <- data.frame(scale(faults[-which(names(faults) %in% depCol)]))
faultLabels <- faults[depCol];

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


## For both datasets, across a range of k, get within-cluster
## sum-of-squared error and average silhouette value.
local({
    fname <- "kmeansClusters.Rda";
    ks <- 2:200;
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
            fAvgLabels <- clusterPredictErr(fc, faultLabels);
            lAvgLabels <- clusterPredictErr(lc, lettersLabels);
            return(data.frame(
                k = k,
                avgS = c(summary(fcSk)$avg.width,
                         summary(lcSk)$avg.width),
                withinSs = c(fc$tot.withinss / nrow(faultsNorm),
                             lc$tot.withinss / nrow(lettersNorm)),
                labelErr = c(sum(fAvgLabels$err) / nrow(faultLabels),
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
});

## Produce a silhouette object to use elsewhere in plots:
local({
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

## Build Mclust models for each dataset.  This will be very
## time-consuming.
local({
    fname <- "emClusters.Rda";
    faultsMcTime <- system.time(
        faultsMc <- Mclust(faultsNorm, 2:500)
    );
    ## The below just seems to loop infinitely even if I greatly
    ## reduce the number of clusters (e.g. 2:3).  I'm not sure what's
    ## going on.
    ## 
    lettersMcTime <- system.time(
        lettersMc <- Mclust(lettersNorm, 2:25)
    );
    faultsBic <- data.frame(
        bic = faultsMc$BIC[,"EII"],
        numClusters = strtoi(names(faultsMc$BIC[,"EII"])),
        test = "Steel faults")
    ##lettersBic <- data.frame(
    ##    bic = lettersMc$BIC[,"EII"],
    ##    numClusters = strtoi(names(lettersMc$BIC[,"EII"])),
    ##    test = "Letters")
    bic <- rbind(faultsBic) ## , lettersBic);
    xlab <- "Number of clusters";
    ylab <- "Bayesian Information Criterion value";
    save(faultsMc, faultsMcTime, xlab, ylab, bic,
         file = fname);
});

###########################################################################
## PCA
###########################################################################

faultsPca <- prcomp(faultsNorm);
lettersPca <- prcomp(lettersNorm);
save(faultsPca, lettersPca, file="pca.Rda");

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

## This might need redone later to compare other reduction techniques
local({
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
});

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

## Reduce the dimensions in PCA:
dims <- 27;
pcaMtx <- pca$rotation[,1:dims];
faultsPca <- as.matrix(faultsNorm) %*% as.matrix(pcaMtx);
clusters <- kmeans(faultsPca, 7, 100);
labelsAvg <- clusterPredictErr(clusters, faultLabels);
sum(labelsAvg$err) / nrow(faultLabels);

dimRange <- 1:27;
kRange <- 2:50;
iters <- 200;
runs <- 100;
t <- system.time(
    faultsPcaSurface <-
        foreach(dims=dimRange, .combine='cbind') %:%
        foreach(k=kRange, .combine='c') %dopar% {
            pcaMtx <- pca$rotation[,1:dims];
            faultsPca <- as.matrix(faultsNorm) %*% as.matrix(pcaMtx);
            clusters <- kmeans(faultsPca, k, iters, runs);
            labelsAvg <- clusterPredictErr(clusters, faultLabels);
            return(sum(labelsAvg$err) / nrow(faultLabels));
        })
print(t);
rownames(faultsPcaSurface) <- kRange;
colnames(faultsPcaSurface) <- dimRange;

save(faultsPcaSurface, file = "faultsPcaSurface.Rda");
persp(faultsPcaSurface, theta = 110, phi = 20, shade=0.6, col="red",
      xlab = "Clusters (k)", ylab = "# of principals",
      zlab = "Error rate");

## faultsPcaDf <- stack(data.frame(t(faultsPcaSurface)));
## faultsPcaDf$pcaDims <- dimRange;
## ggplot(data=faultsPcaDf,
##       aes(x=pcaDims, y=values, group=ind)) +
##    geom_line(aes(colour=ind)) +
##    xlab("Principal component");

ks <- 1:20;
dimRange <- 15:27;
iters <- 50;
runs <- 20;
clusterSs <- foreach(k = ks, .combine='rbind') %:%
    foreach(dims=dimRange, .combine='rbind') %dopar% {
        pcaMtx <- pca$rotation[,1:dims];
        faultsPca <- as.matrix(faultsNorm) %*% as.matrix(pcaMtx);
        clusters <- kmeans(faultsPca, k, iters, runs);
        return(data.frame(k = k,
                          dims = dims,
                          totss = clusters$totss,
                          tot.withinss = clusters$tot.withinss,
                          betweenss = clusters$betweenss));
}

ggplot(data=clusterSs,
       aes(x=k, y=tot.withinss, group=factor(dims))) +
    geom_line(aes(colour=factor(dims)));

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

local({
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
});

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

local({
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
});

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
        data.frame(dims = dims,
                   k = k,
                   err = sum(labelsAvg$err) / nrow(labels));
    }
}

local({
    fname <- "kmeansReducedDims.Rda";
    iters <- 200;
    runs <- 100;
    ##origClusters <- kmeans(faultsNorm, kFaults, iters, runs);

    ks <- c(15, 45, 135, 405);
    
    ## TODO: Add timing information to this!
    faultsPca <- prcomp(faultsNorm);
    pcaKmeansReducedF <-
        foreach(dims=2:27, .combine = "rbind") %dopar% {
            mtxR <- as.matrix(faultsPca$rotation[,1:dims]);
            proj <- as.matrix(faultsNorm) %*% mtxR;
            df <- clusterLabelErrFrame(proj, faultLabels, ks, iters, runs);
            df$test <- "Steel faults";
            df$algo <- "PCA";
        }
    
    dimRange <- 2:26;
    icaKmeansReducedF <-
        foreach(dims=dimRange, .combine = "rbind") %dopar% {
            ica <- fastICA(faultsNorm, dims);
            A <- as.matrix(ica$A);
            S <- as.matrix(ica$S);
            proj <- S %*% A;
            foreach(k=ks, .combine = "rbind") %do% {
                clusters <- kmeans(proj, k, iters, runs);
                labelsAvg <- clusterPredictErr(clusters, faultLabels);
                data.frame(dims = dims,
                           k = k,
                           test = "Steel faults",
                           algo = "ICA",
                           err = sum(labelsAvg$err) / nrow(faultLabels));
            }
        }

    kmeansReducedDims <- rbind(pcaKmeansReducedF,
                               icaKmeansReducedF);
    ## TODO: Add 'k' value to this (though that's not present for EM)
    title <- "Cluster labels on dimension-reduced data";
    
    save(kmeansReducedDims, title, file=fname);
});
