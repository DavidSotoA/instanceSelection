package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._

object Main {
  def main(args: Array[String]): Unit = {
    val spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
    val sc = spark.sqlContext.sparkContext

    val numHashTables = args(0).toInt
    // se cargan y preparan los datos
    val base = spark.read.load(args(1))
    base.createOrReplaceTempView("test")
    val instances = spark.sql("SELECT " + Constants.cols +
      " FROM test WHERE label!=0 and resp_code=1 or resp_code=2")
    val names = instances.columns
    val ignore = Array("idn", "label", "resp_code", "fraude", "nolabel")
    val selectFeatures = for (i <- names if !(ignore contains i )) yield i
    // se realiza el LSH
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val normalizeDF = Mathematics.normalize(vectorizedDF, Constants.SET_OUPUT_COL_ASSEMBLER)
    val randomHyperplanes = new RandomHyperplanes(vectorizedDF, numHashTables, spark)
    val instancesWithSignature = randomHyperplanes.lsh(Constants.SET_OUPUT_COL_SCALED)

    instancesWithSignature.write.mode(SaveMode.Overwrite).format("parquet")
      .save(args(2) + "/prueba")

    // val instancesKeys = LSH.getKeys(instancesWithSignature)
    // se realiza el instance selection
    val drop3 = new Drop3()
    val instanceWithDrop3 = drop3.instanceSelection(instancesWithSignature, true, 2)

    instanceWithDrop3.write.mode(SaveMode.Overwrite).format("parquet").save(args(2) + "/drop3")

  }
}
