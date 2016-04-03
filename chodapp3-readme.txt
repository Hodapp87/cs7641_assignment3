Chris Hodapp (chodapp3), 2016-04-03
Georgia Institute of Technology, CS7641, Machine Learning, Spring 2016
Assignment 3: Unsupervised Learning

This assignment was written in R.  Everything was tested on R v3.2.3
on Linux (64-bit).  It relies on the following CRAN packages:
ggplot2, RSNNS, mclust, fastICA, cluster, doParallel, RPEnsemble, FSelector

The code (minus report & analysis code) is also available at:
https://github.com/Hodapp87/cs7641_assignment3

Short procedure for generating everything:
1. Install packages in R with:
install.packages(c("ggplot2","RSNNS","mclust","fastICA","cluster",
                   "doParallel", "RPEnsemble", "FSelector"));
Or, just source setup.R.
2. Run 'unsupervised_learning.R'.  This will take a long time to run,
possibly days for the Mclust tests.
3. Run 'chodapp3-report.R'.

'unsupervised_learning.R' loads and conditions the data, performs all
of the actual learning/fitting, and generates intermediate files.  All
of the graphs are done in the embedded R code in
'chodapp3-analysis.Rnw', a NoWeb file with LaTeX and R.
'chodapp3-report.R' produces from this file a final PDF, numerous
intermediate PDFs of graphs, LaTeX code, and R code (which will
contain all of the code for generating graphs).

The two data sets are small enough that they are included.  The Steel
Plates Fault Data Set is at
https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults and this
is the source of the files 'Faults.NNA' and 'Faults27x7_var'.  The
Letter Recognition Data Set is at
https://archive.ics.uci.edu/ml/datasets/Letter+Recognition and this is
the source of the file 'letter-recognition.data'.
