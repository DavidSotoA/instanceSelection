package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql._

object Main {
  def main(args: Array[String]): Unit = {
    val spark = Utilities.initSparkSession
    val sc = spark.sqlContext.sparkContext
    val instances = spark.createDataFrame(Seq(
              (0, Vectors.dense(1.0, 3.0), 1, 1),
              (0, Vectors.dense(5.0, -7.0), 2, 1), 
              (0, Vectors.dense(-18.0, -12.0), 3, -1),
              (0, Vectors.dense(-6.0, 31.0), 4, -1),
              (1, Vectors.dense(-61.0, 5.0), 5, 1),
              (1, Vectors.dense(-54.0, 14.0), 6, 1)
            )).toDF("signature", "features", "idn", "label")
    val drop3 = new Drop3()
    val pruebaVentana = drop3.instanceSelection(instances, true)
    pruebaVentana.write.mode(SaveMode.Overwrite).format("parquet")
      .save("/home/david/Escritorio/pruebaVentana")

    /*
    val aggKnn = new AggKnn()

    val instances = spark.createDataFrame(Seq(
              (0, Vectors.dense(1.0, 3.0)),
              (0, Vectors.dense(5.0, -7.0)),
              (0, Vectors.dense(-18.0, -12.0)),
              (0, Vectors.dense(-61.0, 31.0))
            )).toDF("id", "label")

    val pruebaAggKnn = instances.groupBy("id")
                               .agg(aggKnn(instances.col("label")).as("prueba"))

   pruebaAggKnn.write.mode(SaveMode.Overwrite).format("parquet")
     .save("/home/david/Escritorio/prueba")




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
    val randomHyperplanes = new RandomHyperplanes(vectorizedDF, numHashTables, spark)
    val instancesWithSignature = randomHyperplanes.lsh()

    instancesWithSignature.write.mode(SaveMode.Overwrite).format("parquet")
      .save(args(2) + "/prueba")

    // val instancesKeys = LSH.getKeys(instancesWithSignature)
    // se realiza el instance selection
    val instanceWithEntropy = Entropia.instanceSelection(instancesWithSignature, true)

    // instancesKeys.write.mode(SaveMode.Overwrite).format("parquet")
    //   .save(args(2) + "/keys.parquet")

    instanceWithEntropy.write.mode(SaveMode.Overwrite).format("parquet")
      .save(args(2) + "/entropy")*/

  }
}
