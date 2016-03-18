// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 3, Unsupervised Learning (2016-04-03)

// To build without formatted logs (for piping or M-x compile):
// sbt -Dsbt.log.noformat=true compile

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.feature.PCA

import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

import breeze.linalg.{DenseVector => BDV}

import org.apache.log4j.{Level, Logger}


//import com.github.tototoshi.csv._

object UnsupervisedLearning {

  def main(args: Array[String]) {
    
    // TODO: Fix the below to work with what's given in
    // https://spark.apache.org/docs/latest/programming-guide.html#initializing-spark
    val conf = new SparkConf()
      .setAppName("Unsupervised Learning")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    // TODO: Can I just log to a file instead?
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val faults = readSteelFaults(sc)
    val faultsIn = faults.map(_._1)
    val faultsOut = faults.map(_._2)
    val summary: MultivariateStatisticalSummary = Statistics.colStats(faultsIn)
    println("Mean: %s" format summary.mean)
    println("Variance: %s" format summary.variance)

    val mean = new BDV(summary.mean.toArray)
    val stdev = new BDV(summary.variance.toArray).map(Math.sqrt)
    val faultsNormed = faultsIn.map { v =>
      // This is kludge-tastic:
      val v1 = new BDV(v.toArray)
      val v2 = (v1 - mean) / stdev
      Vectors.dense(v2.toArray)
    }

    // Cluster the data into two classes using KMeans
    for (numClusters <- 2 to 40) {
      val numIterations = 100

      val clusters = KMeans.train(faultsNormed, numClusters, numIterations)
      // Evaluate clustering by computing Within Set Sum of Squared Errors
      val WSSSE = clusters.computeCost(faultsNormed)
      println(s"Non-PCA, k = $numClusters: WSSSE = $WSSSE")
      /*
      for (center <- clusters.clusterCenters) {
        println(s"$center")
      }
       */
    }

    for (dims <- 4 to 16) {
      // Attempt PCA
      val pca = new PCA(dims).fit(faultsNormed)
      val pcaMtx = pca.pc
      //println(s"PCA matrix: $pcaMtx")
      val faults2 = pca.transform(faultsNormed)

      for (numClusters <- 2 to 40) {
        val numIterations = 100

        val clusters = KMeans.train(faults2, numClusters, numIterations)
        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = clusters.computeCost(faults2)
        println(s"PCA ($dims dimensions), k = $numClusters: WSSSE = $WSSSE")
      }
    }

    /*
    for (k <- 2 to 10) {
      println(s"k = $k:")
      // Cluster the data into two classes using GaussianMixture
      val gmm = new GaussianMixture().setK(2).run(faults2)

      // Save and load model
      //gmm.save(sc, "myGMMModel")
      //val sameModel = GaussianMixtureModel.load(sc, "myGMMModel")

      // output parameters of max-likelihood model
      for (i <- 0 until gmm.k) {
        println("weight=%f\nmu=%s\nsigma=\n%s\n" format
          (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
      }
    }
     */
    
    sc.stop()
  }

  /** Read all the data from the "Steel Faults" data set, returning
    * an RDD of (inputs, outputs). */
  def readSteelFaults(sc : SparkContext) : RDD[(Vector, Vector)] = {
    // Load and parse the data
    val fname = "/home/hodapp/source/cs7641_assignment3/Faults.NNA"
    val data = sc.textFile(fname).map { str =>
      val vals = str.split('\t').map(_.toDouble)
      (Vectors.dense(vals.slice(0, 27)), Vectors.dense(vals.slice(27, 34)))
    }.cache()
    val rows = data.count()
    println(s"Read $rows rows from $fname.")
    data
  }
}
