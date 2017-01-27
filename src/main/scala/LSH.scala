package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

trait LSH {
  def hashFunction(instance: Vector, hashFunctions: Array[Vector]): Int
  def lsh(): DataFrame
  def groupForBuckets(hashedDataSet: DataFrame): DataFrame
  def keyDistance(x: Vector, y: Vector): Array[Array[Vector]]

  // those methods need are implements here
  // protected def LshKnn(instance: Vector, bucketDataset: Array[Vector],
  // numOfNeighbors: Int): Array[Vector]
  // protected def findBucket(bucketsDataSet: Array[Array[Vector]]): Array[Vector]

}
