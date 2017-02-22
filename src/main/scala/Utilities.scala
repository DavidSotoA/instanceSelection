package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

object Utilities {

  def initSparkContext(): SparkContext = {
    val conf = new SparkConf().setAppName(Constants.APP_NAME).setMaster(Constants.MASTER)
    new SparkContext(conf)
  }

  def initSparkSession(): SparkSession = {
    SparkSession.builder()
     .master(Constants.MASTER)
     .appName(Constants.APP_NAME)
     .config("spark.some.config.option", "some-value")
     .getOrCreate()
  }

  def createVectorDataframe(selectFeatures: Array[String], df: DataFrame): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(selectFeatures)
      .setOutputCol(Constants.SET_OUPUT_COL_ASSEMBLER)

    val vectorDF = assembler.transform(df)
    vectorDF.select("idn", Constants.SET_OUPUT_COL_ASSEMBLER, "label")
  }

}
