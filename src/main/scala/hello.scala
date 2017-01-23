package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = Utilities.initSparkSession
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.0, 3.0)),
      (1, Vectors.dense(1, 6, 7)))
    ).toDF("id", "instance")
    val randomHyperplanes = new RandomHyperplanes(instances, 3)
    val instancesWithSignature = randomHyperplanes.lsh()
    instancesWithSignature.show()
  }
}
