#!/usr/bin/env Rscript

###########################################################################
## Chris Hodapp (chodapp3)
## Georgia Institute of Technology
## CS7641, Machine Learning, Spring 2016
## Assignment 3, Unsupervised Learning (2016-04-03)
###########################################################################

library(ggplot2);
library(RSNNS);
library(mclust);

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

###########################################################################
## k-means
###########################################################################

clusters <- kmeans(faultsNorm, 7, 100);

## Get just the labels, and put the classes alongside:
labels <- faults[depCol];
labels$class <- clusters$cluster;

## Compute squared distance from each instance to the cluster that
## 'owns' it:
sqDist <- data.frame(
    sqDist = rowSums((clusters$center[labels$class,] - faultsNorm)^2),
    class = clusters$cluster);
sqDistSum <- aggregate(sqDist ~ class, sqDist, sum);
bins <- 500;
## In 'breaks' we must include the max(...) or otherwise 'hist' won't
## complete, as values will fall outside the bins.  We set the rest to
## 2*mean because that seems (empirically) to be a useful range, and
## then remove the final bin at the end because it distorts the axis.
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
                     ## Drop the last bin (it's just a catch-all and
                     ## distorts axes):
                     hist = df$sqDist[-bins],
                     class = df$class,
                     bins = mids[-bins]));
distFlat <- do.call(rbind, dtmp);

ggplot(data=distFlat,
       aes(x=bins, y=hist, group=factor(class))) +
    geom_line(aes(colour=factor(class))) +
    xlab("Distance to cluster center") +
    ylab("Frequency") +
    ggtitle("Distance distribution in each cluster")

## Average the labels (as binary vectors) across each class:
labelsAvg <- aggregate(. ~ class, labels, mean);
## Set 'argmax' to the factor that has the highest average:
## (sort of like we did with the neural networks)
labelsAvg$argmax <- factor(apply(labelsAvg[depCol], 1, which.max),
                           labels = depCol, levels = 1:length(depCol));
## Set 'errRate' to the sum of other factors (these are all "wrong" if
## we use the highest factor):
labelsAvg$errRate <- apply(
    labelsAvg[depCol], 1, function(x) (1 - max(x)));
## Then set 'err' to that multiplied by the cluster size to tell us
## error as a number of instances, not as a rate:
labelsAvg$size <- clusters$size[labelsAvg$class];
labelsAvg$err <- labelsAvg$errRate * labelsAvg$size;
## This then gives one metric of error:
sum(labelsAvg$err) / nrow(labels);

###########################################################################
## PCA
###########################################################################
pca <- prcomp(faultsNorm);

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

ggplot(data=contribStacked,
       aes(x=ind, y=values, group=feature)) +
    geom_line(aes(colour=feature))
## Problem: Column 'ind' is not numerical, and it is sorting wrong.
