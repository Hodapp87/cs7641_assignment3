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
library(fastICA);

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

## Also standardize the data to mean 0, variance 1, leaving out the
## labels:
faultsNorm <- data.frame(scale(faults[-which(names(faults) %in% depCol)]))

## Load data for "Letter Recognition" data set & apply headers:
letters <- read.table("letter-recognition.data", sep=",", header=FALSE);
colnames(letters) <- c("Letter", "Xbox", "Ybox", "Width", "Height",
                       "OnPix", "Xbar", "Ybar", "X2bar", "Y2bar",
                       "XYbar", "X2Ybar", "XY2bar", "Xedge",
                       "XedgeXY", "Yedge", "YedgeYX");
## https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
lettersNorm <- data.frame(
    scale(letters[-which(names(letters) == "Letter")]));

labels <- faults[depCol];

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
## corresponding to the data used to generate the clusters, perform
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
        labelsAvg[depCol], 1, function(x) (1 - max(x)));
    ## Then set 'err' to that multiplied by the cluster size to tell
    ## us error as a number of instances, not as a rate:
    labelsAvg$size <- clusters$size[labelsAvg$class];
    labelsAvg$err <- labelsAvg$errRate * labelsAvg$size;
    return(labelsAvg);
}

## Get data for the within-cluster sum-of-squared error, across
## different k, for both datasets.
local({
    fname <- "clusterWithinSs.Rda";
    ks <- 2:100;
    iters <- 100;
    runs <- 50;
    runtime <- system.time(
        clusterWithinSs <- foreach(k = ks, .combine='rbind') %dopar% {
            cat(k);
            cat("..");
            fc <- kmeans(faultsNorm,  k, iters, runs);
            lc <- kmeans(lettersNorm, k, iters, runs);
            return(data.frame(
                k = k,
                faultsWithinSs  = fc$tot.withinss / nrow(faultsNorm),
                lettersWithinSs = lc$tot.withinss / nrow(lettersNorm)));
        }
    )
    print(runtime);
    title <- sprintf("k-means, %d iters, %d runs",
                     iters, runs);
    xlab <- "k (number of clusters)";
    ylab <- "Average squared error";
    save(clusterWithinSs, title, xlab, ylab, runtime, file=fname);
})

p1 <- ggplot(data=faultClusterSs,
             aes(x=k, y=tot.withinss)) +
    geom_line()
## Well, that's not very useful...

ks <- 1:200;
clusterSs <- foreach(k = ks, .combine='rbind') %dopar% {
    clusters <- kmeans(lettersNorm, k, 100, 20);
    return(data.frame(k = k,
                      totss = clusters$totss,
                      tot.withinss = clusters$tot.withinss,
                      betweenss = clusters$betweenss));
}

p2 <- ggplot(data=clusterSs,
             aes(x=k, y=tot.withinss)) +
    geom_line()
multiplot(p1, p2, cols=2)

    
);

clusters <- kmeans(faultsNorm, 30, 100, 20);
heatmap(as.matrix(
    dist(clusters$centers, upper = TRUE, diag = TRUE)));

clustersHist <- distHistogram(clusters, faultsNorm, clusters$cluster);
ggplot(data=clustersHist,
       aes(x=bins, y=hist, group=factor(class))) +
    geom_line(aes(colour=factor(class))) +
    xlab("Distance to cluster center") +
    ylab("Cumulative probability") +
    ggtitle("Distance distribution in each cluster")

## Generate n*n rows, one for each cluster paired with each cluster.
n <- nrow(clusters$centers);
clusterToEach <- clusters$centers[rep(1:n, times=n),];
centersHist <- distHistogram(clusters, clusterToEach, rep(1:n, each=n));
ggplot(data=centersHist,
       aes(x=bins, y=hist, group=factor(class))) +
    geom_line(aes(colour=factor(class))) +
    xlab("Distance to cluster center") +
    ylab("Cumulative probability") +
    ggtitle("Distance between clusters")
## This doesn't really look very good...

labelsAvg <- clusterPredictErr(clusters, labels);

## This then gives one metric of error:
sum(labelsAvg$err) / nrow(labels);

###########################################################################
## EM
###########################################################################
mc <- Mclust(faultsNorm, 50);
summary(mc);

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

lettersPca <- prcomp(lettersNorm);
pcaErrPlot <- data.frame(
    dims = 1:nrow(lettersPca$rotation),
    err = reconstrError(lettersPca$rotation, lettersNorm)
);
ggplot(data=pcaErrPlot,
       aes(x = dims, y = err)) +
    geom_line()


pca <- prcomp(faultsNorm);
pcaErrPlot <- data.frame(
    dims = 1:nrow(pca$rotation),
    err = reconstrError(pca$rotation, faultsNorm)
);
ggplot(data=pcaErrPlot,
       aes(x = dims, y = err)) +
    geom_line()

pcaPlot <- data.frame(pcaStdev = pca$sdev,
                      pcaDim = 1:ncol(faultsNorm));
ggplot(data=pcaPlot,
       aes(x=pcaDim, y=pcaStdev)) +
    geom_line()
## biplot(pca, pc.biplot = TRUE, scale = 0.8);

## Now, this might give some notion of the contribution of each
## original component up to the indication principal:
contrib <- apply(pca$rotation^2, 1, cumsum);
## (That is, row 'i' stands for the i'th principal component and
## column 'j' for the j'th feature, and the value at (i,j) is how much
## feature 'j' has contributed to the first 'i' principals.)
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
labelsAvg <- clusterPredictErr(clusters, labels);
sum(labelsAvg$err) / nrow(labels);

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
            labelsAvg <- clusterPredictErr(clusters, labels);
            return(sum(labelsAvg$err) / nrow(labels));
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
