#!/usr/bin/env Rscript

library(tools);

Sweave("chodapp3-analysis.Rnw");
Stangle("chodapp3-analysis.Rnw");
texi2dvi("chodapp3-analysis.tex", pdf = TRUE);
