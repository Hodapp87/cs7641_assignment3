#!/usr/bin/env Rscript

###########################################################################
## Chris Hodapp (chodapp3)
## Georgia Institute of Technology
## CS7641, Machine Learning, Spring 2016
## Assignment 3, Unsupervised Learning (2016-04-03)
###########################################################################

library(ggplot2);
library(RSNNS);

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
bins <- 40;
breaks <- c(seq(min(sqDist$sqDist), mean(sqDist$sqDist), length=bins),
            max(sqDist$sqDist));
sqDistHist <- aggregate(
    sqDist ~ class,
    sqDist,
    function(x) {
        h <- hist(x, breaks = breaks, plot = FALSE)
        ## Drop the last bin (it's just a catch-all):
        return(h$density[-bins]);
    },
    simplify = FALSE);
## This just passes dummy data to get the midpoints:
mids <- hist(breaks, breaks = breaks, plot = FALSE)$mids[-bins]
h <- apply(sqDistHist,
   1,
   function(df) data.frame(
                    hist = df$sqDist,
                    class = df$class,
                    bins = mids));
do.call(rbind, h);

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
