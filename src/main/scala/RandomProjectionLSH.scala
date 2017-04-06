package com.lsh

import scala.util.Random

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

case class RandomProjectionLSH(
    dataset_RH: Dataset[_],
    andsFunctions: Int,
    orsFunctions: Int,
    spark_RH: SparkSession) extends LSH {

    dataset = dataset_RH
    spark = spark_RH
    require(andsFunctions > 0, "andsFunctions debe ser mayor a cero")
    require(orsFunctions > 0, "orsFunctions debe ser mayor a cero")
    val functionFamilies = createFunctionsForFamilies(orsFunctions, andsFunctions )

    def createFunctions(inputDim: Int): Vector = {
      Vectors.dense(Array.fill(inputDim)(Random.nextGaussian()))
    }

    def createFunctionsForFamilies(
      orsFunctions: Int,
      andsFunctions: Int): Seq[(Int, List[(Vector, Float)])] = {
      var functionsForFamilies = Seq[(Int, List[(Vector, Float)])]()
      val inputDim = dataset.select(Constants.SET_OUPUT_COL_ASSEMBLER).head.get(0)
        .asInstanceOf[Vector].size
      for(i <- 1 to orsFunctions) {
        var w = 0
        do {
          w = Random.nextInt(100) + 1
        }while(functionsForFamilies.exists(_._1 == w))

        var functions = List[(Vector, Float)]()
        for(i <- 1 to andsFunctions){
          val a = createFunctions(inputDim)
          val b = Random.nextFloat * (w - 1) + 1
          functions = functions :+ (a, b)
        }
        functionsForFamilies = functionsForFamilies :+ (w, functions)
      }
      functionsForFamilies
    }

    def getFamilies(): Seq[(Int, List[(Vector, Float)])] = {
      functionFamilies
    }

    def hashFunction(instance: Vector, hashFunctions: Array[Vector]): String = {
      throw new IllegalArgumentException ("unimplement method")
    }

    def lsh(colForLsh: String): DataFrame = {
      throw new IllegalArgumentException ("unimplement method")
    }

    def keyDistance(x: Vector, y: Vector): Array[Array[Vector]] = {
      throw new IllegalArgumentException ("unimplement method")
    }
}
