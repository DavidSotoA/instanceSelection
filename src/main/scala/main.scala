package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = Utilities.initSparkSession
    val numHashTables = args(0).toInt
    val base = spark.read.load("/home/skorpionx/test")
    base.createOrReplaceTempView("test")
    val instances = spark.sql("SELECT "+ Constants.cols+ " FROM test WHERE label!=0 and resp_code=1 or resp_code=2")
    val names = instances.columns
    val ignore = Array("idn", "label","resp_code","fraude","nolabel")
    val selectFeatures = for (i <- names if !(ignore contains i )) yield i
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val randomHyperplanes = new RandomHyperplanes(vectorizedDF, numHashTables, spark)
    val instancesWithSignature = randomHyperplanes.lsh()
    val instancesKeys = LSH.getKeys(instancesWithSignature)
    instancesWithSignature.write.mode(SaveMode.Overwrite).format("parquet").save("/home/skorpionx/Escritorio/prueba.parquet")
    instancesKeys.write.mode(SaveMode.Overwrite).format("parquet").save("/home/skorpionx/Escritorio/keys.parquet")
  }
}
