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

  test("Se calcula la entropia correctamente") {
    val instances1 = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (0, Vectors.dense(-1.0, 1.0))
    )).toDF(Constants.LABEL, "keys")

    val instances2 = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (0, Vectors.dense(1.0, -1.0)),
      (0, Vectors.dense(-1.0, -1.0)),
      (0, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    val instances3 = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (1, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

   assert(Entropia.calcularEntropia(instances1) == 1.0)
   assert(Entropia.calcularEntropia(instances2) == 0.0)
   assert(Entropia.calcularEntropia(instances3) == 0.8112781244591328)
  }

  test("Se seleccionan todas las muestras cuando la entropia es 1") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (1, Vectors.dense(-1.0, -1.0)),
      (0, Vectors.dense(-1.0, 1.0))
    )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances).count == 4.0)
  }

  test("Se seleccionan todas las muestras cuando la entropia es 0") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (0, Vectors.dense(1.0, -1.0)),
      (0, Vectors.dense(-1.0, -1.0)),
      (0, Vectors.dense(-1.0, 1.0))
      )).toDF(Constants.LABEL, "keys")

    assert(Entropia.instanceSelection(instances).count == 1.0)
  }

  test("Se seleccionan todas las muestras cuando la entropia es distinta de 0 y 1") {
      val instances = spark.createDataFrame(Seq(
        (0, Vectors.dense(1.0, 1.0)),
        (1, Vectors.dense(1.0, -1.0)),
        (0, Vectors.dense(-1.0, -1.0)),
        (1, Vectors.dense(-1.0, -1.0)),
        (1, Vectors.dense(-1.0, -1.0)),
        (0, Vectors.dense(-1.0, -1.0)),
        (1, Vectors.dense(-1.0, -1.0)),
        (1, Vectors.dense(-1.0, -1.0)),
        (1, Vectors.dense(-1.0, -1.0)),
        (1, Vectors.dense(-1.0, 1.0))
      )).toDF("label", "keys")

    assert(Entropia.instanceSelection(instances).count == 9)
  }


}
