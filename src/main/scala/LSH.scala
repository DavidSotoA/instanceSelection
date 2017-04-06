package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._

trait LSH {
  var dataset: Dataset[_] = _
  var numHashTables: Int = _
  var spark: SparkSession = _

  def hashFunction(instance: Vector, hashFunctions: Array[Vector]): String
  def lsh(colForLsh: String): DataFrame
  def keyDistance(x: Vector, y: Vector): Array[Array[Vector]]

  def groupForBuckets(hashedDataSet: DataFrame): DataFrame = {
    hashedDataSet.sort(Constants.SET_OUPUT_COL_LSH)
  }
}

object LSH{
  def getKeys(hashedDataSet: DataFrame): DataFrame = {
    hashedDataSet.select(Constants.SET_OUPUT_COL_LSH).distinct()
  }

  def findBucket(bucketsDataSet: DataFrame, key: String): DataFrame = {
    bucketsDataSet.select("*")
      .where(Constants.SET_OUPUT_COL_LSH + " == '" + key + "'")
  }
}
