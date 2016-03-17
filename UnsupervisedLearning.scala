// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 3, Unsupervised Learning (2016-04-03)

// To build without formatted logs (for piping or M-x compile):
// sbt -Dsbt.log.noformat=true compile

/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel

import com.github.tototoshi.csv._

object SimpleApp {
  def main(args: Array[String]) {
    val logFile = "kmeans_data.txt"

    // TODO: Fix the below to work with what's given in
    // https://spark.apache.org/docs/latest/programming-guide.html#initializing-spark
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
    kmeans(sc)
    gaussianMixtures(sc)
    sc.stop()
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
