package com.lsh

import scala.util.Random

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

case class RandomProjectionLSH(
    dataset_RH: Dataset[_],
    andsFunctions: Int,
    orsFunctions: Int,
    sizeBucket: Double,
    spark_RH: SparkSession) extends LSH {

    dataset = dataset_RH
    spark = spark_RH
    require(andsFunctions > 0, "andsFunctions debe ser mayor a cero")
    require(orsFunctions > 0, "orsFunctions debe ser mayor a cero")
    var functionFamilies = createFunctionsForFamilies(orsFunctions, andsFunctions )

    def createFunctions(inputDim: Int): Vector = {
      Mathematics.normalizeVector(
        Vectors.dense(Array.fill(inputDim)(Random.nextDouble())))
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

    def hashFunction2(instance: Vector, familieId: String, familieFunctions: List[(Vector, Double)]): String = {
      val signature = familieFunctions.map {
        case (a, b) => ((Mathematics.dot(a, instance) + b) / sizeBucket).floor.toInt
      }
      // familieId + Mathematics.stringSignature(signature.toArray)
      familieId + Mathematics.stringSignature(signature.toArray)
    }

    def hashFunction(instance: Vector, hashFunctions: Array[Vector]): String = {
      throw new IllegalArgumentException ("unimplement method")
    }

    def lsh(colForLsh: String): DataFrame = {
      var udfs = List[org.apache.spark.sql.expressions.UserDefinedFunction]()
      var i = 0
      while(i < functionFamilies.size) {
        val j = i
        val familie = functionFamilies(i)
        val partiallyHashFunction = hashFunction2( _ : Vector, j.toString, familie)
        udfs = udfs :+ udf(partiallyHashFunction)
        i = i + 1
      }
      var j = 0
      var act_udf = udfs(j)
      var signatureDF = dataset.withColumn((Constants.SET_OUPUT_COL_LSH),
                    act_udf(dataset(colForLsh)))
      j = j + 1
      while(j < functionFamilies.size) {
        act_udf = udfs(j)
        val signatureFamilie = dataset.withColumn((Constants.SET_OUPUT_COL_LSH),
                      act_udf(dataset(colForLsh)))
        signatureDF = signatureDF.union(signatureFamilie)
        j = j + 1
      }
      signatureDF
    }

    def keyDistance(x: Vector, y: Vector): Array[Array[Vector]] = {
      throw new IllegalArgumentException ("unimplement method")
    }
}
