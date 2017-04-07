package com.lsh

import scala.util.Random

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

case class RandomProjectionLSH(
    dataset_RH: Dataset[_],
    andsFunctions: Int,
    orsFunctions: Int,
    sizeBucket: Int,
    spark_RH: SparkSession) extends LSH {

    dataset = dataset_RH
    spark = spark_RH
    require(andsFunctions > 0, "andsFunctions debe ser mayor a cero")
    require(orsFunctions > 0, "orsFunctions debe ser mayor a cero")
    var functionFamilies = createFunctionsForFamilies(orsFunctions, andsFunctions )

    def createFunctions(inputDim: Int): Vector = {
      Vectors.dense(Array.fill(inputDim)(Random.nextGaussian()))
    }

    def createFunctionsForFamilies(
      orsFunctions: Int,
      andsFunctions: Int): Seq[(List[(Vector, Double)])] = {
      var functionsForFamilies = Seq[List[(Vector, Double)]] ()
      val inputDim = dataset.select(Constants.SET_OUPUT_COL_ASSEMBLER).head.get(0)
        .asInstanceOf[Vector].size
      for(i <- 1 to orsFunctions) {
        var functions = List[(Vector, Double)]()
        for(i <- 1 to andsFunctions){
          val a = createFunctions(inputDim)
          val b = Random.nextDouble * (sizeBucket - 1) + 1
          functions = functions :+ (a, b)
        }
        functionsForFamilies = functionsForFamilies :+ (functions)
      }
      functionsForFamilies
    }

    def setFamilies(families: Seq[List[(Vector, Double)]]) {
      functionFamilies = families
    }

    def getFamilies(): Seq[List[(Vector, Double)]] = {
      functionFamilies
    }

    def hashFunction2(instance: Vector, familieFunctions: List[(Vector, Double)]): String = {
      val signature = familieFunctions.map {
        case (a, b) => ((Mathematics.dot(a, instance) + b) / sizeBucket).floor.toInt
      }
      Mathematics.stringSignature(signature.toArray)
    }

    def hashFunction(instance: Vector, hashFunctions: Array[Vector]): String = {
      throw new IllegalArgumentException ("unimplement method")
    }

    def lsh(colForLsh: String): DataFrame = {
      var i = 1
      var signatureDF = dataset
      for(familie <- functionFamilies) {
        val partiallyHashFunction = hashFunction2( _ : Vector, familie)
        val transformUDF = udf(partiallyHashFunction)
        signatureDF = signatureDF.withColumn((Constants.SET_OUPUT_COL_LSH + "_" + i),
                      transformUDF(signatureDF(colForLsh)))
        i = i +1
      }
      signatureDF.toDF
    }

    def keyDistance(x: Vector, y: Vector): Array[Array[Vector]] = {
      throw new IllegalArgumentException ("unimplement method")
    }
}
