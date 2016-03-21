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

clusters <- kmeans(faultsNorm, 20, 20);
## f <- fitted(clusters, "classes");
labels <- faults[depCol];
labels$class <- clusters$cluster;
labelsAvg <- aggregate(. ~ class, labels, mean);
labelsAvg$size <- clusters$size[labelsAvg$class];
