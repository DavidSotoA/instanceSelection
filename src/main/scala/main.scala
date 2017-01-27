package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = Utilities.initSparkSession

    val numHashTables = 3
    val selectFeatures = Array("c1", "c2", "c3", "c4", "c5", "c6")
    val instances = spark.createDataFrame(Seq(
      (4, 234, 1344, 5, 345, 123),
      (0, 123, 4356, 135, 567, 1823),
      (789, 1523, 556, 7865, 3485, 1283),
      (3, 1783, 56, 5, 345, 123),
      (4, 5464, 4, 578, 7852, 45))
    ).toDF("c1", "c2", "c3", "c4", "c5", "c6")
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val randomHyperplanes = new RandomHyperplanes(vectorizedDF, numHashTables, spark)
    val instancesWithSignature = randomHyperplanes.lsh()
    instancesWithSignature.write.format("parquet").save("/home/skorpionx/Escritorio/lsh.parquet")
  }
}
