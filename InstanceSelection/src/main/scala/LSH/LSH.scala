package lsh

import utilities.Constants
import mathematics.Mathematics
import org.apache.spark.sql.Row
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf

trait LSH {
  var dataset: DataFrame = _
  var numHashTables: Int = _
  var spark: SparkSession = _

  def hashFunction(instance: Vector, hashFunctions: Array[Vector]): String
  def lsh(colForLsh: String): DataFrame
}

object Lsh{
  def getKeys(hashedDataSet: DataFrame): DataFrame = {
    hashedDataSet.select(Constants.COL_SIGNATURE).distinct()
  }

  def findBucket(bucketsDataSet: DataFrame, key: String): DataFrame = {
    bucketsDataSet.select("*")
      .where(Constants.COL_SIGNATURE + " == '" + key + "'")
  }

  def subBuckets(maxBucketSize: Int, df: DataFrame, sc: SparkContext): DataFrame = {
    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val countBucketDf = df.groupBy(Constants.COL_SIGNATURE).count
    val bigBuckets = countBucketDf.select("*").where(s"count >  $maxBucketSize")
    if (bigBuckets.rdd.isEmpty) {
      return df
    }
    val roulette = (maxBucketSize: Int, bucketSize: Int ) => 1.toDouble/math.ceil(bucketSize.toDouble/maxBucketSize)
    val rouletteUdf = udf(roulette(maxBucketSize, _ : Int))
    val rouletteDf = bigBuckets.withColumn(Constants.COL_ROULETTE, rouletteUdf(bigBuckets("count"))).drop("count")
    val bucketsForDivide = rouletteDf.collect.map(x => (x(0).asInstanceOf[String], x(1).asInstanceOf[Double])).toSeq

    val spinRouletteWithList = spinRoulette(_ : String, bucketsForDivide)
    return df.map(x =>(x(0).asInstanceOf[Int],
                x(1).asInstanceOf[Vector],
                x(2).asInstanceOf[Int],
                spinRouletteWithList(x(3).asInstanceOf[String]))).toDF(Constants.COL_ID, Constants.COL_FEATURES,
                      Constants.COL_LABEL, Constants.COL_SIGNATURE)

  }

  def spinRoulette(signature: String, list: Seq[(String, Double)]): String = {
    if (!list.exists(_._1 == signature)) {
      return signature
    }
    var (_, rouletteChunk) = list.find(_._1 == signature).get
    var rnd = scala.util.Random.nextFloat
    val newValueInSignature = Math.ceil(rnd/rouletteChunk).toInt
    return signature + newValueInSignature.toString
  }

  def lsh(
    method: String,
    instances: DataFrame,
    spark: SparkSession,
    sizeBucket: Double = 1,
    andFunctions: Int,
    orFunctions: Int = 1): DataFrame = {
      val normalizeDF = Mathematics.normalize(instances, Constants.COL_FEATURES)
      method match {
        case Constants.LSH_HYPERPLANES_METHOD => {
          val randomHyperplanes = new RandomHyperplanes(normalizeDF, andFunctions, spark)
          return randomHyperplanes.lsh(Constants.COL_SCALED).drop(Constants.COL_SCALED)
        }
        case Constants.LSH_PROJECTION_METHOD => {
          val randomProjection = new RandomProjectionLSH(normalizeDF, andFunctions, orFunctions, sizeBucket, spark)
          return randomProjection.lsh(Constants.COL_SCALED).drop(Constants.COL_SCALED)
        }
        case _ => throw new IllegalArgumentException("El método " + method + " no existe")
      }
    }
}
