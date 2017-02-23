package com.test

import com.lsh.AggKnn
import com.lsh.Constants
import com.lsh.Drop3
import com.lsh.Mathematics
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

class Drop3Test extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll() {
    spark = Utilities.initSparkSession
  }

  test("Se calcula la distancia entre 2 muestras") {
    val v1 = Vectors.dense(1, 3, 5)
    val v2 = Vectors.dense(13, -63, -4)
    assert(Mathematics.distance(v1, v2) == 67.68308503607086)
    assert(Mathematics.distance(v2, v1) == 67.68308503607086)
  }

  test("Se eliminan los amigos mas cercanos") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0), 0),
      (1, Vectors.dense(1.0, -1.0), 1),
      (2, Vectors.dense(-1.0, -1.0), 0),
      (3, Vectors.dense(-1.0, -1.0), 0),
      (4, Vectors.dense(-1.0, -1.0), 0),
      (5, Vectors.dense(-1.0, -1.0), 1),
      (6, Vectors.dense(-1.0, -1.0), 1),
      (7, Vectors.dense(-1.0, -1.0), 0),
      (8, Vectors.dense(-1.0, -1.0), 1),
      (9, Vectors.dense(-1.0, 1.0), 0)
    )).toDF("id", "keys", "label")


    val instance = instances.head
    assert(Drop3.killFriends(instance, instances).count == 4)
  }

  test("Se arroja IllegalArgumentException si el numero de vecinos no es positivo") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0), 0),
      (1, Vectors.dense(1.0, -1.0), 1),
      (2, Vectors.dense(-1.0, -1.0), 0),
      (3, Vectors.dense(-1.0, -1.0), 0),
      (4, Vectors.dense(-1.0, -1.0), 0),
      (5, Vectors.dense(-1.0, -1.0), 1),
      (6, Vectors.dense(-1.0, -1.0), 1),
      (7, Vectors.dense(-1.0, -1.0), 0),
      (8, Vectors.dense(-1.0, -1.0), 1),
      (9, Vectors.dense(-1.0, 1.0), 0)
    )).toDF("id", "features", "label")

    val v = Vectors.dense(13, -63, -4)
    assertThrows[IllegalArgumentException]{
      Drop3.knn(v, instances, 0)
    }
  }

  test("El metodo knn devuelve un dataframe con la estructura correcta1") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0), 0),
      (1, Vectors.dense(1.0, -1.0), 1),
      (2, Vectors.dense(-1.0, -1.0), 0),
      (3, Vectors.dense(-1.0, -1.0), 0),
      (4, Vectors.dense(-1.0, -1.0), 0),
      (5, Vectors.dense(-1.0, -1.0), 1),
      (6, Vectors.dense(-1.0, -1.0), 1),
      (7, Vectors.dense(-1.0, -1.0), 0),
      (8, Vectors.dense(-1.0, -1.0), 1),
      (9, Vectors.dense(-1.0, 1.0), 0)
    )).toDF("id", "features", "label")
    val v = Vectors.dense(1, 4)
    val columnNames = Drop3.knn(v, instances, 3).schema.fieldNames
    assert(columnNames.length == 2)
    assert(columnNames(0) == "id")
    assert(columnNames(1) == "distance")

  }

  test("Se hallan los k vecinos mas cercanos") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0), 0),
      (1, Vectors.dense(1.0, -1.0), 1),
      (2, Vectors.dense(-1.0, -1.0), 0),
      (3, Vectors.dense(-1.0, -1.0), 0),
      (4, Vectors.dense(-1.0, -1.0), 0),
      (5, Vectors.dense(-1.0, -1.0), 1),
      (6, Vectors.dense(-1.0, -1.0), 1),
      (7, Vectors.dense(-1.0, -1.0), 0),
      (8, Vectors.dense(-1.0, -1.0), 1),
      (9, Vectors.dense(-1.0, 1.0), 0)
    )).toDF("id", "features", "label")
    val v = Vectors.dense(1, 4)
    val knnDF = Drop3.knn(v, instances, 3)
    val ids = knnDF.select("id").collect

    assert(ids(0)(0).asInstanceOf[Int] == 0)
    assert(ids(1)(0).asInstanceOf[Int] == 9)
    assert(ids(2)(0).asInstanceOf[Int] == 1)
  }

  test("La funcion de agregacion AggKnn devuelve un array de tipo vector con todas las"
        + " caracteristicas de todas las muestras") {
          val aggKnn = new AggKnn()

          val instances = spark.createDataFrame(Seq(
                    (0, Vectors.dense(1.0, 3.0)),
                    (0, Vectors.dense(5.0, -7.0)),
                    (0, Vectors.dense(-18.0, -12.0)),
                    (0, Vectors.dense(-61.0, 31.0))
                  )).toDF("signature", "features")

          val pruebaAggKnn = instances.groupBy("signature")
                                     .agg(aggKnn(instances.col("features")).as("prueba"))
          val resp = pruebaAggKnn.head
          assert(resp(1).asInstanceOf[Seq[Vector]].length == 4)
  }

  test("Al apicar las ventanas cada instancia queda con una nueva columna que contiene "
    + "todas las instancias de la cubeta") {
          val instances = spark.createDataFrame(Seq(
                    (0, Vectors.dense(1.0, 3.0)),
                    (0, Vectors.dense(5.0, -7.0)),
                    (0, Vectors.dense(-18.0, -12.0)),
                    (0, Vectors.dense(-6.0, 31.0)),
                    (1, Vectors.dense(-61.0, 5.0)),
                    (1, Vectors.dense(-54.0, 14.0))
                  )).toDF("signature", "features")
          val drop3 = new Drop3()
          val pruebaVentana = drop3.instanceSelection(instances, true)

          val c1 = pruebaVentana.select("colOfDistances").where("signature == 0")
          val c2 = pruebaVentana.select("colOfDistances").where("signature == 1")

          val respc1 = c1.head
          val respc2 = c2.head

          assert(respc1(0).asInstanceOf[Seq[Vector]].length == 4)
          assert(respc2(0).asInstanceOf[Seq[Vector]].length == 2)
  }

}
