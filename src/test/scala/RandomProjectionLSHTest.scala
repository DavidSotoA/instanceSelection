package com.test

import com.lsh.Constants
import com.lsh.LSH
import com.lsh.Mathematics
import com.lsh.RandomProjectionLSH
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

class RandomProjectionLSHTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _
  var randomHyperplanes: RandomProjectionLSH = _
  var numHashTables: Int = _

  override def beforeAll() {
    spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
  }

  test("Se retorna IllegalArgumentException si el nuero de andsFunctions es menor o igual a cero"){
    val andsFunctions = 0
    val orsFunctions = 3
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.0, 3.0)),
      (1, Vectors.dense(1, 6, 7)))
    ).toDF("id", "instance")

    assertThrows[IllegalArgumentException]{
      val hyp = new RandomProjectionLSH(instances, andsFunctions, orsFunctions,   spark)
    }
  }

  test("Se retorna IllegalArgumentException si el nuero de orsFunctions es menor o igual a cero"){
    val andsFunctions = 1
    val orsFunctions = 0
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.0, 3.0)),
      (1, Vectors.dense(1, 6, 7)))
    ).toDF("id", "instance")

    assertThrows[IllegalArgumentException]{
      val hyp = new RandomProjectionLSH(instances, andsFunctions, orsFunctions,   spark)
    }
  }

  test("Se crean las familias de funciones") {
    val selectFeatures = Array("c2", "c3", "c4", "c5")
    val instances = spark.createDataFrame(Seq(
      (4, 234, 1344, 5, 345, 123),
      (0, 123, 4356, 135, 567, 1823),
      (789, 1523, 556, 7865, 3485, 1283),
      (3, 1783, 56, 5, 345, 123),
      (4, 5464, 4, 578, 7852, 45))
    ).toDF("idn", "c2", "c3", "c4", "c5", "label")

    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val andsFunctions = 2
    val orsFunctions = 2
    val rp = new RandomProjectionLSH(vectorizedDF, andsFunctions, orsFunctions,   spark)
    val functionFamilies = rp.getFamilies
    assert(functionFamilies.size == orsFunctions)
    assert(functionFamilies(0)._2.size == andsFunctions)
  }

}
