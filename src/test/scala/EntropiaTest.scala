package com.test

import com.lsh.Constants
import com.lsh.Entropia
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

class EntropiaTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll() {
    spark = Utilities.initSparkSession
  }

  test("Se seleccionan todas las muestras cuando la entropia es 1 y las muestras debalanceadas") {
    val instances = spark.createDataFrame(Seq(
      (-1, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, 1.0))
    )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, true).count == 2.0)
  }

  test("Se seleccionan todas las muestras cuando la entropia es 1 y las muestras balanceadas") {
    val instances = spark.createDataFrame(Seq(
      (-1, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, 1.0))
    )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, false).count == 4.0)
  }

  test("Se seleccionan todas las muestras cuando la entropia es 0, todas las muestras son de la clase minoritaria(desbalanceado)") {
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      ( 1, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, true).count == 4.0)
  }

  test("Se seleccionan todas las muestras cuando la entropia es 0, todas las muestras son de la clase mayoritaria(desbalanceado)") {
    val instances = spark.createDataFrame(Seq(
      (-1, Vectors.dense(1.0, 1.0)),
      (-1, Vectors.dense(1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, true).count == 1.0)
  }

  test("Se seleccionan todas las muestras cuando la entropia es 0(balanceado)") {
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (1, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, false).count == 1.0)
  }

  test("Se seleccionan las muestras cuando la entropia no es 0 ni uno y las clases estan desbalanceadas") {
    val instances = spark.createDataFrame(Seq(
      (-1, Vectors.dense(1.0, 1.0)),
      (-1, Vectors.dense(1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, true).count == 3.0)
  }

  test("Se seleccionan las muestras cuando la entropia no es 0 ni uno y las clases estan balanceadas") {
    val instances = spark.createDataFrame(Seq(
      (-1, Vectors.dense(1.0, 1.0)),
      (-1, Vectors.dense(1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (-1, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances, false).count == 5.0)
  }
}
