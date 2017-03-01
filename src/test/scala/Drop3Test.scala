package com.test

import com.lsh.AggKnn
import com.lsh.Constants
import com.lsh.Drop3
import com.lsh.Mathematics
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

class Drop3Test extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll() {
    spark = Utilities.initSparkSession
  }

  test("Se arroja IllegalArgumentException si el numero de vecinos es par"){
    val instances = spark.createDataFrame(Seq(
              (0, Vectors.dense(1.0, 3.0), 1),
              (0, Vectors.dense(5.0, -7.0), 2),
              (0, Vectors.dense(-18.0, -12.0), 3),
              (0, Vectors.dense(-6.0, 31.0), 4),
              (1, Vectors.dense(-61.0, 5.0), 5),
              (1, Vectors.dense(-54.0, 14.0), 6)
            )).toDF("signature", "features", "idn")
    val drop3 = new Drop3()
    assertThrows[IllegalArgumentException]{
      drop3.instanceSelection(instances, true, 2)
    }
  }

  test("Se arroja IllegalArgumentException si el numero de vecinos es negativo"){
    val instances = spark.createDataFrame(Seq(
              (0, Vectors.dense(1.0, 3.0), 1),
              (0, Vectors.dense(5.0, -7.0), 2),
              (0, Vectors.dense(-18.0, -12.0), 3),
              (0, Vectors.dense(-6.0, 31.0), 4),
              (1, Vectors.dense(-61.0, 5.0), 5),
              (1, Vectors.dense(-54.0, 14.0), 6)
            )).toDF("signature", "features", "idn")
    val drop3 = new Drop3()
    assertThrows[IllegalArgumentException]{
      drop3.instanceSelection(instances, true, -2)
    }
  }

  test("Se calculan las distancias y se ordenan descendentemente para una instancia dada"){
    val instances = Row(Seq(
      Row(Vectors.dense(9,8), 3, 1),
      Row(Vectors.dense(1,2), 5, -1)
    ))
    val instance = Vectors.dense(1,1)
    val drop3 = new Drop3()
    val distances = drop3.calculateDistances(instance, instances(0).asInstanceOf[Seq[Row]])
    assert(distances(0) == (1.0, 5, -1))
    assert(distances(1) == (10.63014581273465, 3, 1))

  }

  test("Se encuentran los k vecinos mas cercanos en base a las distancias calculadas"){
    val instances = Seq( (1.0, 3, 1), (5.0, 3, 1), (2.0, 3, 1), (20.0, 3, 1),
                          (456.0, 3, 1), (100.0, 3, 1))
    val instance = Vectors.dense(1,1)
    val drop3 = new Drop3()
    val neighbors = drop3.findNeighbors(instances, 3, true)
    assert(neighbors(0) == (1.0, 3, 1))
    assert(neighbors(1) == (2.0, 3, 1))
    assert(neighbors(2) == (5.0, 3, 1))
  }

  test("Se eliminan las intancias que tienen el mismo label") {
    val instances = Seq( (1.0, 3, 1), (5.0, 3, -1), (2.0, 3, -1), (20.0, 3, 1),
                          (456.0, 3, 1), (100.0, 3, 1))
    val drop3 = new Drop3()
    val enemies = drop3.killFriends(instances, 1)
    assert(enemies.size == 2)
  }

  test("Se encuentra al enemigo mas cercano") {
    val instances = Seq( (1.0, 3, 1), (5.0, 3, -1), (2.0, 3, -1), (20.0, 3, 1),
                          (456.0, 3, 1), (100.0, 3, 1))
    val drop3 = new Drop3()
    val nemseis = drop3.findMyNemesis(instances, 1,true)
    assert(nemseis == 2.0)
  }
}
