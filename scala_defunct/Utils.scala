package cs7641

// Spark:
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import breeze.linalg.{DenseVector => BDV}

// NetCDF:
/*
import ucar.nc2.{NetcdfFile, NetcdfFileWriter, Dimension, Variable}
import ucar.ma2.DataType
import ucar.ma2.{Array => netcdfArray}
 */

package object Utils {

  /** Normalize the input vectors to be mean 0 and variance 1 (at every
    * dimension) */
  def normalize(in : RDD[Vector]) : RDD[Vector] = {
    // This is kludge-tastic:
    val s: MultivariateStatisticalSummary = Statistics.colStats(in)
    val mean = new BDV(s.mean.toArray)
    val stdev = new BDV(s.variance.toArray).map(Math.sqrt)
    in.map { v =>
      val v1 = new BDV(v.toArray)
      val v2 = (v1 - mean) / stdev
      Vectors.dense(v2.toArray)
    }
  }



}
