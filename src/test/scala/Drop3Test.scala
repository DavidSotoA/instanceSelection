package com.test

import com.github.martincooper.datatable.{DataColumn, DataRow, DataTable, DataValue}
import com.lsh.AggKnn
import com.lsh.Constants
import com.lsh.Drop3
import com.lsh.Mathematics
import com.lsh.RowTable
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession


class Drop3Test extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll() {
    spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
  }

  test("Se arroja IllegalArgumentException si el numero de vecinos es par") {
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

  test("Se arroja IllegalArgumentException si el numero de vecinos es negativo") {
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

  test("Se calculan las distancias y se ordenan descendentemente para una instancia dada") {
    val instances = Row(Seq(
      Row(Vectors.dense(9, 8), 3, 1),
      Row(Vectors.dense(1, 2), 5, -1)
    ))
    val instance = Vectors.dense(1, 1)
    val drop3 = new Drop3()
    val distances = drop3.calculateDistances(instance, instances(0).asInstanceOf[Seq[Row]])
    assert(distances(0) == (1.0, 5, -1))
    assert(distances(1) == (10.63014581273465, 3, 1))

  }

  test("Se encuentran los k vecinos mas cercanos en base a las distancias calculadas") {
    val instances = Seq( (1.0, 3, 1), (5.0, 3, 1), (2.0, 3, 1), (20.0, 3, 1),
                          (456.0, 3, 1), (100.0, 3, 1))
    val instance = Vectors.dense(1, 1)
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
    val nemesis = drop3.findMyNemesis(instances, 1, true)
    assert(nemesis == 2.0)
  }

  test("Se crea el datatable inicial") {
    val instances = Seq(Row(1.0, 3, 1), Row(5.0, 4, -1),
                        Row(2.0, 5, -1), Row(20.0, 6, 1))
    val drop3 = new Drop3()
    val table = drop3.createDataTable(instances)
    assert(table.size == 4)
    assert(table(0)(0) == (3, 1))
    assert(table(1)(0) == (4, -1))
    assert(table(2)(0) == (5, -1))
    assert(table(3)(0) == (6, 1))
  }

  test("Dado un datatable y un id, se obtiene el indice y el row correpondiente a dicho id") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(20.0), 6, 1))
    val drop3 = new Drop3()
    val table = drop3.createDataTable(instances)
    val (indice, row) = drop3.getIndexAndRowById(5, table)
    val waitedRow = new RowTable((5, -1), null, null, 0.0, List())
    assert(indice == 2)
    assert(row.isInstanceOf[RowTable])
    assert(row == waitedRow)
  }

  test("se llena el datatable segun la lista dada") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, 1))
    val drop3 = new Drop3()
    val table = drop3.completeTable(instances, 3, drop3.createDataTable(instances))
    val row0: IndexedSeq[Any] = Vector((3, 1),
                                       Seq((1.0, 5, -1), (2.0, 6, -1), (4.0, 4, -1), (19.0, 7, 1)),
                                       Seq((1.0, 5, -1), (2.0, 6, -1), (4.0, 4, -1), (19.0, 7, 1)),
                                       1.0,
                                       List(4, 5, 6, 7))
    val row1: IndexedSeq[Any] = Vector((4, -1),
                                       Seq((2.0, 6, -1), (3.0, 5, -1), (4.0, 3, 1), (15.0, 7, 1)),
                                       Seq((2.0, 6, -1), (3.0, 5, -1), (4.0, 3, 1), (15.0, 7, 1)),
                                       4.0,
                                       List(3, 5, 6, 7))
    val row2: IndexedSeq[Any] = Vector((5, -1),
                                      Seq((1.0, 3, 1), (1.0, 6, -1), (3.0, 4, -1), (18.0, 7, 1)),
                                      Seq((1.0, 3, 1), (1.0, 6, -1), (3.0, 4, -1), (18.0, 7, 1)),
                                      1.0,
                                      List(3, 4, 6, 7))
    val row3: IndexedSeq[Any] = Vector((6, -1),
                                       Seq((1.0, 5, -1), (2.0, 3, 1), (2.0, 4, -1), (17.0, 7, 1)),
                                       Seq((1.0, 5, -1), (2.0, 3, 1), (2.0, 4, -1), (17.0, 7, 1)),
                                       2.0,
                                       List(3, 4, 5, 7))
    val row4: IndexedSeq[Any] = Vector((7, 1),
                                       Seq((15.0, 4, -1), (17.0, 6, -1), (18.0, 5, -1),
                                       (19.0, 3, 1)),
                                       Seq((15.0, 4, -1), (17.0, 6, -1), (18.0, 5, -1),
                                       (19.0, 3, 1)),
                                       15.0,
                                       List(3, 4, 5, 6))
    assert(table.size == 5)
    assert(table(0).values == row0)
    assert(table(1).values == row1)
    assert(table(2).values == row2)
    assert(table(3).values == row3)
    assert(table(4).values == row4)
  }
}
