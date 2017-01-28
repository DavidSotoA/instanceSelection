package com.lsh

import scala.util.Random

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

case class RandomHyperplanes(
    dataset_RH: Dataset[_],
    numHashTables_RH: Int,
    spark_RH: SparkSession) extends LSH{
  dataset = dataset_RH
  numHashTables = numHashTables_RH
  spark = spark_RH
  require(numHashTables > 0, "numHashTables debe ser mayor a cero")
  var hyperplanes = createHiperplanes()

  def getHyperplanes(): Array[Vector] = {
    hyperplanes
  }

  def setHyperplanes(set_hyperplanes: Array[Vector]){
      hyperplanes = set_hyperplanes
  }

  def createHiperplanes(): Array[Vector] = {
    val inputDim = dataset.select(Constants.SET_OUPUT_COL_ASSEMBLER).head.get(0)
      .asInstanceOf[Vector].size
    Array.fill(numHashTables) {
      Vectors.dense(Array.fill(inputDim)(Random.nextGaussian()))
    }
  }

  override def lsh(): DataFrame = {
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

  override def keyDistance(x: Vector, y: Vector): Array[Array[Vector]] = {
    throw new IllegalArgumentException ("unimplement method")
  }

}
