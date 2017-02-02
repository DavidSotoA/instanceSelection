package com.test

import com.lsh.Constants
import com.lsh.LSH
import com.lsh.Mathematics
import com.lsh.RandomHyperplanes
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

class RandomHyperplanesTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _
  var randomHyperplanes: RandomHyperplanes = _
  var numHashTables: Int = _

  override def beforeAll() {
    spark = Utilities.initSparkSession
    numHashTables = 3
    val selectFeatures = Array("c2", "c3", "c4", "c5")
    val instances = spark.createDataFrame(Seq(
      (4, 234, 1344, 5, 345, 123),
      (0, 123, 4356, 135, 567, 1823),
      (789, 1523, 556, 7865, 3485, 1283),
      (3, 1783, 56, 5, 345, 123),
      (4, 5464, 4, 578, 7852, 45))
    ).toDF("idn", "c2", "c3", "c4", "c5", "label")
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    randomHyperplanes = new RandomHyperplanes(vectorizedDF, numHashTables, spark)
  }

  test("Arrojar IllegalArgumentException si se da un numero de numHashTables negativo o cero") {
    val numHashTablesFail = 0
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.0, 3.0)),
      (1, Vectors.dense(1, 6, 7)))
    ).toDF("id", "instance")

    assertThrows[IllegalArgumentException]{
      val hyp = new RandomHyperplanes(instances, numHashTablesFail, spark)
    }
  }

  test("Se crean el numero de hiperplanos correctos") {
    val hyperplanes = randomHyperplanes.createHiperplanes()
    assert(numHashTables == hyperplanes.length)
  }

  test("Los hiperplanos tienen el numero correcto de caracteristicas") {
    val numFeatures = 4
    val hyperplanes = randomHyperplanes.createHiperplanes()
    assert(numFeatures == hyperplanes(0).size)
  }

  test("Se realiza el producto punto") {
    val x = Vectors.dense(1.0, 0.5, 3.0)
    val y = Array(Vectors.dense(4.0, -4.0, 1.0), Vectors.dense(-4.0, -4.0, 1.0),
      Vectors.dense(6.0, -4.0, 1.0), Vectors.dense(4.0, -14.0, 1.0))
    assert(Mathematics.dot(x, y(0)) == 5)
    assert(Mathematics.dot(x, y(1)) == -3)
    assert(Mathematics.dot(x, y(2)) == 7)
    assert(Mathematics.dot(x, y(3)) == 0)
  }

  test("Se convierte de binario a decimal") {
    val numBin = Array(Array(1, 0, 1, 1), Array(1, 1, 0, 1, 0, 1, 1),
      Array(0, 0, 0, 0), Array(1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1))
    assert(Mathematics.binaryToDec(numBin(0)) == 11)
    assert(Mathematics.binaryToDec(numBin(1)) == 107)
    assert(Mathematics.binaryToDec(numBin(2)) == 0)
    assert(Mathematics.binaryToDec(numBin(3)) == 13765)
  }

  test("Se realiza la hashfunction") {
    val x = Vectors.dense(1.0, 0.5, 3.0)
    val hyperlanes = Array(Vectors.dense(4.0, -4.0, 1.0), Vectors.dense(-4.0, -4.0, 1.0),
      Vectors.dense(6.0, -4.0, 1.0), Vectors.dense(4.0, -14.0, 1.0))
    val signature = randomHyperplanes.hashFunction(x, hyperlanes)
    assert(signature == "1011")
  }

  test("El metodo lsh retorna un dataframe con la estructura correcta") {
      val instancesWithSignature = randomHyperplanes.lsh()
      val columnNames = instancesWithSignature.schema.fieldNames
      assert(instancesWithSignature.count() == 5)
      assert(columnNames.length == 4)
      assert(columnNames(0) == "idn")
      assert(columnNames(1) == Constants.SET_OUPUT_COL_ASSEMBLER)
      assert(columnNames(2) == "label")
      assert(columnNames(3) == Constants.SET_OUPUT_COL_LSH)
    }

  test("El metodo lsh calcula las firmas correctas") {
    val selectFeatures = Array("c1", "c2")
    val instances = spark.createDataFrame(Seq(
      (0, 3.0, 0.5, 0),
      (1, 4.0, 0.4, 0),
      (2, -0.5, 3.0, 1),
      (3, -0.4, 4.0, 0),
      (4, -0.5, -3.0, 1),
      (5, -0.4, -4.0, 0))
    ).toDF("idn", "c1", "c2", "label")
    val hyperplanes = Array(
      Vectors.dense(-1, 1),
      Vectors.dense(0, 1),
      Vectors.dense(1, 1),
      Vectors.dense(1, 0)
    )
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val hyp = new RandomHyperplanes(vectorizedDF, 4, spark)
    hyp.setHyperplanes(hyperplanes)
    val instancesWithSignature = hyp.lsh()
    val keysDF = LSH.getKeys(instancesWithSignature)
    val size = keysDF.count
    val keys = keysDF.select(Constants.SET_OUPUT_COL_LSH).take(3)
    assert(size == 3)
    assert(keys(0)(0).asInstanceOf[String] == "0000")
    assert(keys(1)(0).asInstanceOf[String] == "0111")
    assert(keys(2)(0).asInstanceOf[String] == "1110")
  }

  test("Se hace la conversion con VectorAssembler") {
    val selectFeatures = Array("c2", "c3", "c4", "c5")
    val instances = spark.createDataFrame(Seq(
      (4, 234, 1344, 5, 345, 123),
      (0, 123, 4356, 135, 567, 1823),
      (789, 1523, 556, 7865, 3485, 1283),
      (3, 1783, 56, 5, 345, 123),
      (4, 5464, 4, 578, 7852, 45))
    ).toDF("idn", "c2", "c3", "c4", "c5", "label")

    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)
    val columnNames = vectorizedDF.schema.fieldNames
    val dimFeatures = vectorizedDF.select(Constants.SET_OUPUT_COL_ASSEMBLER)
      .head.get(0).asInstanceOf[Vector].size
    assert(columnNames.length == 3)
    assert(columnNames(0) == "idn")
    assert(columnNames(1) == Constants.SET_OUPUT_COL_ASSEMBLER)
    assert(columnNames(2) == "label")
    assert(dimFeatures == selectFeatures.length)
  }

}
