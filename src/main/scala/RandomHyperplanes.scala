package com.lsh

import scala.util.Random

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

case class RandomHyperplanes(dataset: Dataset[_], numHashTables: Int ) extends LSH{
  require(numHashTables > 0, "numHashTables debe ser mayor a cero")

  def createHiperplanes(inputDim: Int): Array[Vector] = {
    Array.fill(numHashTables) {
      Vectors.dense(Array.fill(inputDim)(Random.nextGaussian()))
    }
  }

  override def lsh(): DataFrame = {
    val inputDim = dataset.select("instance").head.get(0).asInstanceOf[Vector].size
    val hyperplanes = createHiperplanes(inputDim)
    val partiallyHashFunction = hashFunction( _ : Vector, hyperplanes)
    val transformUDF = udf(partiallyHashFunction)
    dataset.withColumn("signature", transformUDF(dataset("instance")))
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

  override def groupForBuckets(hashedDataSet: Array[Vector]): Array[Array[Vector]] = {
    throw new IllegalArgumentException ("unimplement method")
  }

  override def keyDistance(x: Vector, y: Vector): Array[Array[Vector]] = {
    throw new IllegalArgumentException ("unimplement method")
  }

}
