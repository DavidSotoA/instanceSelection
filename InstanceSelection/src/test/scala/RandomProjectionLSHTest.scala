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
      val hyp = new RandomProjectionLSH(instances, andsFunctions, orsFunctions, 3, spark)
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
      val hyp = new RandomProjectionLSH(instances, andsFunctions, orsFunctions, 3, spark)
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
    val rp = new RandomProjectionLSH(vectorizedDF, andsFunctions, orsFunctions, 3, spark)
    val functionFamilies = rp.getFamilies
    assert(functionFamilies.isInstanceOf[Seq[(List[(Vector, Float)])]])
    assert(functionFamilies.size == orsFunctions)
    assert(functionFamilies(0).size == andsFunctions)
  }

  test("Dada una familia se crea la firma de una muestra") {
    val selectFeatures = Array("c2", "c3")
    val instances = spark.createDataFrame(Seq(
      (4, 23,5 , -1),
      (0, 12, 43, 1),
      (789, 15, 55, -1),
      (3, 17, 56, 1),
      (4, 54, 4, 1))
    ).toDF("idn", "c2", "c3", "label")

    val instance = Vectors.dense(23,5 )
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val andsFunctions = 2
    val orsFunctions = 2
    val rp = new RandomProjectionLSH(vectorizedDF, andsFunctions, orsFunctions, 3, spark)
    val familieFunctions = List((Vectors.dense(0.1, 0.2), 1.0),
                                 (Vectors.dense(0.3, 0.002), 2.0),
                                 (Vectors.dense(0.8, 0.78), 0.67))

   val signature = rp.hashFunction2(instance, "1", familieFunctions)
   assert(signature == "1127")
  }

  test("se normalizan los datos") {
    val x = Vectors.dense(1.5, 0.5 , 7.8, 6.7)
    val xNormalized = Mathematics.normalizeVector(x)
    assert(xNormalized == Vectors.dense(0.75, 0.25, 3.9, 3.35))
  }

  test("El metodo lsh calcula las firmas correctas") {
    val selectFeatures = Array("c1", "c2")
    val instances = spark.createDataFrame(Seq(
      (0, 3.0, 0.5, -1),
      (1, 4.0, 0.4, -1),
      (2, -0.5, 3.0, 1),
      (3, -0.4, 4.0, -1),
      (4, -0.5, -3.0, 1),
      (5, -0.4, -4.0, -1))
    ).toDF("idn", "c1", "c2", "label")

    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val andsFunctions = 2
    val orsFunctions = 2
    val rp = new RandomProjectionLSH(vectorizedDF, andsFunctions, orsFunctions, 3, spark)
    val familieFunctions = Seq(List((Vectors.dense(0.1, 0.2), 1.0),
                                     (Vectors.dense(0.3, 0.002), 2.0)),
                               List((Vectors.dense(0.567, 0.89), 1.4),
                                     (Vectors.dense(- 0.22, 0.98), 2.9)),
                               List((Vectors.dense(0.6637, 0.289), 1.4),
                                    (Vectors.dense(- 0.56, 0.198), 2.9))
                                   )
    rp.setFamilies(familieFunctions)
    val lsh = rp.lsh(Constants.SET_OUPUT_COL_ASSEMBLER)
    val x = lsh.collect
    // familie 1
    assert(x(0)(3).asInstanceOf[String] == "000")
    assert(x(1)(3).asInstanceOf[String] == "001")
    assert(x(2)(3).asInstanceOf[String] == "000")
    assert(x(3)(3).asInstanceOf[String] == "000")
    assert(x(4)(3).asInstanceOf[String] == "000")
    assert(x(5)(3).asInstanceOf[String] == "000")

    // familie 2
    assert(x(6)(3).asInstanceOf[String] == "110")
    assert(x(7)(3).asInstanceOf[String] == "110")
    assert(x(8)(3).asInstanceOf[String] == "111")
    assert(x(9)(3).asInstanceOf[String] == "112")
    assert(x(10)(3).asInstanceOf[String] == "1-10")
    assert(x(11)(3).asInstanceOf[String] == "1-1-1")

    // familie 3
    assert(x(12)(3).asInstanceOf[String] == "210")
    assert(x(13)(3).asInstanceOf[String] == "210")
    assert(x(14)(3).asInstanceOf[String] == "201")
    assert(x(15)(3).asInstanceOf[String] == "201")
    assert(x(16)(3).asInstanceOf[String] == "200")
    assert(x(17)(3).asInstanceOf[String] == "2-10")

  }
}
