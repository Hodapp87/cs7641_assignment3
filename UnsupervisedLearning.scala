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

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.feature.PCA

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel

import org.apache.log4j.{Level, Logger}

//import com.github.tototoshi.csv._

object UnsupervisedLearning {

  def main(args: Array[String]) {
    
    // TODO: Fix the below to work with what's given in
    // https://spark.apache.org/docs/latest/programming-guide.html#initializing-spark
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    // TODO: Can I just log to a file instead?
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val faults = readSteelFaults(sc)

    // Attempt PCA
    val dims = 15
    val pca = new PCA(dims).fit(faults)
    val pcaMtx = pca.pc
    println(s"PCA matrix: $pcaMtx")
    val faults2 = pca.transform(faults)

    // Cluster the data into two classes using KMeans
    for (numClusters <- 2 to 40) {
      println(s"k = $numClusters:")
      
      val numIterations = 20
      val clusters = KMeans.train(faults2, numClusters, numIterations)

      // Evaluate clustering by computing Within Set Sum of Squared Errors
      val WSSSE = clusters.computeCost(faults2)
      println("Within Set Sum of Squared Errors = " + WSSSE)
      for (center <- clusters.clusterCenters) {
        println(s"$center")
      }
    }

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
    
    /*
    val logFile = "kmeans_data.txt"

    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
    kmeans(sc)
    gaussianMixtures(sc)*/
    sc.stop()
  }

  // Major TODO: This ignores the labels.
  /** Read all the data from the "Steel Faults" data set. */
  def readSteelFaults(sc : SparkContext) : RDD[Vector] = {
    // Load and parse the data
    val fname = "Faults.NNA"
    val data = sc.textFile(fname)
    val parsedData = data.map(
      s => Vectors.dense(s.split('\t').map(_.toDouble))).cache()

    val rows = parsedData.count()
    println(s"Read $rows rows from $fname.")

    parsedData
  }

  def kmeans(sc: SparkContext) {

    // Load and parse the data
    val data = sc.textFile("kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
      println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    //clusters.save(sc, "myModelPath")
    //val sameModel = KMeansModel.load(sc, "myModelPath")
  }

  def gaussianMixtures(sc: SparkContext) {

    // Load and parse the data
    val data = sc.textFile("gmm_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using GaussianMixture
    val gmm = new GaussianMixture().setK(2).run(parsedData)

    // Save and load model
    //gmm.save(sc, "myGMMModel")
    //val sameModel = GaussianMixtureModel.load(sc, "myGMMModel")

    // output parameters of max-likelihood model
    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }
  }
}
