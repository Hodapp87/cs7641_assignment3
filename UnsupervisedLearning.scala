// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 3, Unsupervised Learning (2016-04-03)

// To build without formatted logs (for piping or M-x compile):
// sbt -Dsbt.log.noformat=true compile

// Running spark-shell with this:
// spark-shell --driver-class-path /home/hodapp/source/cs7641_assignment3/target/scala-2.10/classes

package cs7641

// Local dependencies:
import cs7641.Utils._

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.{RDD, DoubleRDDFunctions}

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

import breeze.linalg.{DenseVector => BDV}
import org.apache.log4j.{Level, Logger}

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

    val faultsNormed = normalize(faultsIn)

    // Cluster the data into two classes using KMeans
    for (numClusters <- List(5, 10, 20, 40, 80)) {

      // Train K-Means & output some basic information:
      val numIterations = 100
      val model : KMeansModel = KMeans.train(faultsNormed, numClusters, numIterations)
      val WSSSE = model.computeCost(faultsNormed)
      println(s"k = $numClusters: WSSSE = $WSSSE")

      // Determine a cluster index for each instance:
      val clusters : RDD[Int] = model.predict(faultsNormed)

      // And then group these together with the class labels:
      val classes : Array[(Int, RDD[Vector])] =
        clusters.zip(faultsOut).groupBy(_._1).collect.map {
          // We have at this stage (Int, Iterable[(Int, Vector)]),
          // which is a little clumsy and redundant.  We don't need
          // the IDs inside the Iterable, and we also want to turn it
          // to an RDD (hence the .collect above - we cannot nest
          // RDDs).
          case (id, iter) => (id, sc.parallelize(iter.map(_._2).toSeq))
        }

      classes.foreach { case(id,v) =>
        println("%s, count %d, %s" format (id, v.count, Statistics.colStats(v).mean))
      }

      // Compute & output a histogram of these:
      /*
      val hist = clusters.countByValue()
      val classes = hist.keys.toList.sorted
      classes.foreach { k =>
        val count = hist(k)
        println(s"Class $k, $count elements")
      }*/
      /*
      for (center <- clusters.clusterCenters) {
        println(s"$center")
      }*/
    }

    for (dims <- 4 to 16) {
      // Attempt PCA
      val pca = new PCA(dims).fit(faultsNormed)
      val pcaMtx = pca.pc
      //println(s"PCA matrix: $pcaMtx")
      val faults2 = pca.transform(faultsNormed)
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
