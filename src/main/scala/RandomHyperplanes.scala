package com.lsh

import scala.util.Random

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

case class RandomHyperplanes(
    dataset: Dataset[_],
    numHashTables: Int,
    spark: SparkSession) extends LSH{
  require(numHashTables > 0, "numHashTables debe ser mayor a cero")

  def createHiperplanes(inputDim: Int): Array[Vector] = {
    Array.fill(numHashTables) {
      Vectors.dense(Array.fill(inputDim)(Random.nextGaussian()))
    }
  }

  override def lsh(): DataFrame = {
    val inputDim = dataset.select(Constants.SET_OUPUT_COL_ASSEMBLER).head.get(0)
      .asInstanceOf[Vector].size
    val hyperplanes = createHiperplanes(inputDim)
    val partiallyHashFunction = hashFunction( _ : Vector, hyperplanes)
    val transformUDF = udf(partiallyHashFunction)
    val signatureDF = dataset.withColumn(Constants.SET_OUPUT_COL_LSH,
      transformUDF(dataset(Constants.SET_OUPUT_COL_ASSEMBLER)))
    groupForBuckets(signatureDF)
  }

  override def hashFunction(instance: Vector,
    hashFunctions: Array[Vector]): Int = {
     val signature = (dotRestult: Double) => {
       if (dotRestult >= 0) {
         1
       } else {
         0
       }
     }
     val binSignature = hashFunctions.map(hash => signature(Utilities.dot(hash, instance)))
     Utilities.binaryToDec(binSignature)
  }

  override def groupForBuckets(hashedDataSet: DataFrame): DataFrame = {
    hashedDataSet.createGlobalTempView("hashedDataSet")
    spark.sql("SELECT * FROM global_temp.hashedDataSet ORDER BY "
      + Constants.SET_OUPUT_COL_LSH)
  }

  override def keyDistance(x: Vector, y: Vector): Array[Array[Vector]] = {
    throw new IllegalArgumentException ("unimplement method")
  }

}
