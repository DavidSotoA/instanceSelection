package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._

trait LSH {
  var dataset: Dataset[_] = _
  var numHashTables: Int = _
  var spark: SparkSession = _

  def hashFunction(instance: Vector, hashFunctions: Array[Vector]): Int
  def lsh(): DataFrame
  def keyDistance(x: Vector, y: Vector): Array[Array[Vector]]

  def groupForBuckets(hashedDataSet: DataFrame): DataFrame = {
    hashedDataSet.createGlobalTempView("hashedDataSet")
    spark.sql("SELECT * FROM global_temp.hashedDataSet ORDER BY "
      + Constants.SET_OUPUT_COL_LSH)
  }

  // those methods need are implements here
  // protected def LshKnn(instance: Vector, bucketDataset: Array[Vector],
  // numOfNeighbors: Int): Array[Vector]
  // protected def findBucket(bucketsDataSet: Array[Array[Vector]]): Array[Vector]

}
