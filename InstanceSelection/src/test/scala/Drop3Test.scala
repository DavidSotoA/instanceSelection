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

/*
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
  }*/

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

  test("Se eliminan las instancias que tienen el mismo label") {
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
    val table = drop3.createDataTable2(instances).getTable
    assert(table.size == 4)
    assert(table(0).id == (3, 1))
    assert(table(1).id == (4, -1))
    assert(table(2).id == (5, -1))
    assert(table(3).id == (6, 1))
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

  test("Se llena el datatable segun la lista dada") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, 1),
                        Row(Vectors.dense(14.0), 8, 1),
                        Row(Vectors.dense(30.0), 9, -1),
                        Row(Vectors.dense(7.0), 10, 1))
    val drop3 = new Drop3()
    val table = drop3.completeTable2(instances, 3, drop3.createDataTable2(instances)).getTable

    val row1 = new RowTable((7,1),
                                      Seq((17.0,6,-1), (18.0,5,-1), (19.0,3,1)),
                                      Seq((6.0,8,1), (10.0,9,-1), (13.0,10,1),
                                      (15.0,4,-1)),
                                      10.0,
                                      List(8, 9))

    val row0 = new RowTable((9,-1),
                                       Seq((27.0,6,-1), (28.0,5,-1), (29.0,3,1)),
                                       Seq((10.0,7,1), (16.0,8,1), (23.0,10,1), (25.0,4,-1)),
                                       10.0,
                                       List(7))

    val row2 = new RowTable((8,1),
                                       Seq((12.0,5,-1), (13.0,3,1), (16.0,9,-1)),
                                       Seq((6.0,7,1), (7.0,10,1), (9.0,4,-1), (11.0,6,-1)),
                                       9.0,
                                       List(7, 9))

    val row5 = new RowTable((4,-1),
                                       Seq((9.0,8,1), (15.0,7,1), (25.0,9,-1)),
                                       Seq((2.0,6,-1), (2.0,10,1), (3.0,5,-1), (4.0,3,1)),
                                       2.0,
                                       List(3, 5, 6, 7, 8, 9, 10))

    val row4 = new RowTable((6,-1),
                                       Seq((11.0,8,1), (17.0,7,1), (27.0,9,-1)),
                                       Seq((1.0,5,-1), (2.0,3,1), (2.0,4,-1), (4.0,10,1)),
                                       2.0,
                                       List(3, 4, 5, 8, 10))

    val row3 = new RowTable((10,1),
                                       Seq((7.0,8,1), (13.0,7,1), (23.0,9,-1)),
                                       Seq((2.0,4,-1), (4.0,6,-1), (5.0,5,-1), (6.0,3,1)),
                                       2.0,
                                       List(3, 4, 5, 6, 7, 8, 9))

    val row7 = new RowTable((3,1),
                                       Seq((13.0,8,1), (19.0,7,1), (29.0,9,-1)),
                                       Seq((1.0,5,-1), (2.0,6,-1), (4.0,4,-1), (6.0,10,1)),
                                       1.0,
                                       List(4, 5, 6, 10))


    val row6 = new RowTable((5,-1),
                                       Seq((12.0,8,1), (18.0,7,1), (28.0,9,-1)),
                                       Seq((1.0,3,1), (1.0,6,-1), (3.0,4,-1), (5.0,10,1)),
                                       1.0,
                                       List(3, 4, 6, 10))

    assert(table.size == 8)
    assert(table(0) == row0)
    assert(table(1) == row1)
    assert(table(2) == row2)
    assert(table(3) == row3)
    assert(table(4) == row4)
    assert(table(5) == row5)
    assert(table(6) == row6)
    assert(table(7) == row7)
  }

  test("El metodo knn retorna la etiqueta que tiene la mayoria de los vecinos") {
    val labels = Seq(-1, 1)
    val neighbors = Seq((2.2, 4, 1), (2.2, 4, 1), (2.2, 4, -1))
    val drop3 = new Drop3()
    val label = drop3.knn(labels, neighbors)
    assert(label == 1)
  }

  test("Se determina si remover una instancia en base a la clasificacion de"
    + " los asosiados de una instancia p con y sin p") {
      val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                          Row(Vectors.dense(5.0), 4, -1),
                          Row(Vectors.dense(2.0), 5, -1),
                          Row(Vectors.dense(3.0), 6, -1),
                          Row(Vectors.dense(20.0), 7, 1))
      val drop3 = new Drop3()
      val table = drop3.completeTable(instances, 3, drop3.createDataTable(instances))
      val remove = drop3.removeInstance((3, 1), List(4, 5, 6, 7), table, List())

      assert(remove == true)
  }

  test("Se actualiza el datatable al remover una instancia") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, 1),
                        Row(Vectors.dense(14.0), 8, 1),
                        Row(Vectors.dense(30.0), 9, -1),
                        Row(Vectors.dense(7.0), 10, 1))
    val drop3 = new Drop3()
    val table = drop3.completeTable(instances, 3, drop3.createDataTable(instances))
    val updateTable = drop3.updateTableForRemove(10, List(3, 4, 5, 6, 7, 8, 9), table, List())

    val row0: IndexedSeq[Any] = Vector((7,1),
                                        Seq((18.0,5,-1), (19.0,3,1)),
                                        Seq((6.0,8,1), (10.0,9,-1), (15.0,4,-1), (17.0,6,-1)),
                                        10.0,
                                        List(8, 9))

    val row1: IndexedSeq[Any] = Vector((9,-1),
                                       Seq((28.0,5,-1), (29.0,3,1)),
                                       Seq((10.0,7,1), (16.0,8,1), (25.0,4,-1), (27.0,6,-1)),
                                       10.0,
                                       List(7))

    val row2: IndexedSeq[Any] = Vector((8,1),
                                       Seq((13.0,3,1), (16.0,9,-1)),
                                       Seq((6.0,7,1), (9.0,4,-1), (11.0,6,-1), (12.0,5,-1)),
                                       9.0,
                                       List(7, 9))

    val row3: IndexedSeq[Any] = Vector((4,-1),
                                       Seq((15.0,7,1), (25.0,9,-1)),
                                       Seq((2.0,6,-1), (3.0,5,-1), (4.0,3,1), (9.0,8,1)),
                                       2.0,
                                       List(3, 5, 6, 7, 8, 9))

    val row4: IndexedSeq[Any] = Vector((6,-1),
                                       Seq((17.0,7,1), (27.0,9,-1)),
                                       Seq((1.0,5,-1), (2.0,3,1), (2.0,4,-1), (11.0,8,1)),
                                       2.0,
                                       List(3, 4, 5, 8, 7, 9))

    val row5: IndexedSeq[Any] = Vector((10,1),
                                       Seq((7.0,8,1), (13.0,7,1), (23.0,9,-1)),
                                       Seq((2.0,4,-1), (4.0,6,-1), (5.0,5,-1), (6.0,3,1)),
                                       2.0,
                                       List(3, 4, 5, 6, 7, 8, 9))

    val row6: IndexedSeq[Any] = Vector((3,1),
                                       Seq((19.0,7,1), (29.0,9,-1)),
                                       Seq((1.0,5,-1), (2.0,6,-1), (4.0,4,-1), (13.0,8,1)),
                                       1.0,
                                       List(4, 5, 6))

    val row7: IndexedSeq[Any] = Vector((5,-1),
                                       Seq((18.0,7,1), (28.0,9,-1)),
                                       Seq((1.0,3,1), (1.0,6,-1), (3.0,4,-1), (12.0,8,1)),
                                       1.0,
                                       List(3, 4, 6, 8))
     assert(table.size == 8)
     assert(updateTable(0).values == row0)
     assert(updateTable(1).values == row1)
     assert(updateTable(2).values == row2)
     assert(updateTable(3).values == row3)
     assert(updateTable(4).values == row4)
     assert(updateTable(5).values == row5)
     assert(updateTable(6).values == row6)
     assert(updateTable(7).values == row7)
  }

  test("Se realiza el metodo de DROP3 con clases balanceadas") {
    val instances = Seq(Row(Vectors.dense(9.0), 3, 1),
                        Row(Vectors.dense(9.5), 4, 1),
                        Row(Vectors.dense(9.8), 5, 1),
                        Row(Vectors.dense(2.0), 6, 1),
                        Row(Vectors.dense(3.0), 7, 1),
                        Row(Vectors.dense(10.34), 8, -1),
                        Row(Vectors.dense(10.2), 9, -1),
                        Row(Vectors.dense(10.1), 10, -1),
                        Row(Vectors.dense(22.1), 11, -1),
                        Row(Vectors.dense(18.1), 12, -1))

    val drop3 = new Drop3()
    val instanceToRemove = drop3.drop3(instances, false, 3)
    assert(instanceToRemove == List(11, 12, 6, 7, 3, 4, 5, 10))
  }

  test("Se realiza el metodo de DROP3 con clases desbalanceadas") {
    val instances = Seq(Row(Vectors.dense(9.0), 3, -1),
                        Row(Vectors.dense(9.5), 4, 1),
                        Row(Vectors.dense(9.8), 5, 1),
                        Row(Vectors.dense(2.0), 6, 1),
                        Row(Vectors.dense(3.0), 7, 1),
                        Row(Vectors.dense(10.34), 8, -1),
                        Row(Vectors.dense(10.2), 9, -1),
                        Row(Vectors.dense(10.1), 10, -1),
                        Row(Vectors.dense(22.1), 11, -1),
                        Row(Vectors.dense(18.1), 12, -1))

    val drop3 = new Drop3()
    val instanceToRemove = drop3.drop3(instances, true, 4)
    assert(instanceToRemove == List(11, 12))
  }

  test("Se realiza el drop3 sobre un dataframe") {
    val instances = spark.createDataFrame(Seq(
              (0, Vectors.dense(1.0, 3.0), 1, 1),
              (0, Vectors.dense(5.0, -7.0), 2, 1),
              (0, Vectors.dense(-18.0, -12.0), 3, 1),
              (0, Vectors.dense(-6.0, 31.0), 4, -1),
              (1, Vectors.dense(-61.0, 5.0), 5, -1),
              (1, Vectors.dense(-54.0, 14.0), 6, -1)
            )).toDF("signature", "features", "idn", "label" )

    val resp = Seq(
      Row(1,0,Vectors.dense(1.0,3.0),1),
      Row(2,0,Vectors.dense(5.0,-7.0),1),
      Row(3,0,Vectors.dense(-18.0,-12.0),1),
      Row(5,1,Vectors.dense(-61.0,5.0),-1))

    val drop3 = new Drop3()
    val prueba = drop3.instanceSelection(instances, true, 3)
    val pruebaCollect = prueba.collect
    assert(resp(0) == pruebaCollect(0))
    assert(resp(1) == pruebaCollect(1))
    assert(resp(2) == pruebaCollect(2))
    assert(resp(3) == pruebaCollect(3))
  }

  test("Se determina que las muestras dadas son todas de una misma clase") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, -1),
                        Row(Vectors.dense(14.0), 8, -1),
                        Row(Vectors.dense(30.0), 9, -1),
                        Row(Vectors.dense(7.0), 10, -1))
    val drop3 = new Drop3()
    val label = instances.head.getInt(2)
    val oneClasss = drop3.isOneClass(instances, label)
    assert(oneClasss == true)
  }

  test("Se determina que las muestras dadas son de diferentes clases") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, -1),
                        Row(Vectors.dense(14.0), 8, -1),
                        Row(Vectors.dense(30.0), 9, -1),
                        Row(Vectors.dense(7.0), 10, 1))
    val drop3 = new Drop3()
    val label = instances.head.getInt(2)
    val oneClasss = drop3.isOneClass(instances, label)
    assert(oneClasss == false)
  }

  test("Si las muestras son desbalanceadas y todas las muestras son de la clase mayoritaria se"
      + " indica que se deben de eliminar todas las instancias menos una") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, -1),
                        Row(Vectors.dense(14.0), 8, -1),
                        Row(Vectors.dense(30.0), 9, -1),
                        Row(Vectors.dense(7.0), 10, -1))
    val drop3 = new Drop3()
    val label = instances.head.getInt(2)
    val unbalanced = true
    val selectInstances = drop3.returnIfOneClass(instances, unbalanced, label)
    assert(selectInstances.size == 7)
  }

  test("Si las muestras son desbalanceadas y todas las muestras son de la clase minoritaria se"
      + " indica que no se deben de eliminar instancias") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, 1),
                        Row(Vectors.dense(2.0), 5, 1),
                        Row(Vectors.dense(3.0), 6, 1),
                        Row(Vectors.dense(20.0), 7, 1),
                        Row(Vectors.dense(14.0), 8, 1),
                        Row(Vectors.dense(30.0), 9, 1),
                        Row(Vectors.dense(7.0), 10, 1))
    val drop3 = new Drop3()
    val label = instances.head.getInt(2)
    val unbalanced = true
    val selectInstances = drop3.returnIfOneClass(instances, unbalanced, label)
    assert(selectInstances.size == 0)
  }

  test("Si las muestras son balanceadas y todas las muestras son de la clase mayoriatria se"
      + " indica que se deben de eliminar todas las instancias menos una") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, 1),
                        Row(Vectors.dense(2.0), 5, 1),
                        Row(Vectors.dense(3.0), 6, 1),
                        Row(Vectors.dense(20.0), 7, 1),
                        Row(Vectors.dense(14.0), 8, 1),
                        Row(Vectors.dense(30.0), 9, 1),
                        Row(Vectors.dense(7.0), 10, 1))
    val drop3 = new Drop3()
    val label = instances.head.getInt(2)
    val unbalanced = false
    val selectInstances = drop3.returnIfOneClass(instances, unbalanced, label)
    assert(selectInstances.size == 7)
  }

  test("Si las muestras son balanceadas y todas las muestras son de la clase minoritaria se"
      + " indica que se deben de eliminar todas las instancias menos una") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, 1),
                        Row(Vectors.dense(2.0), 5, 1),
                        Row(Vectors.dense(3.0), 6, 1),
                        Row(Vectors.dense(20.0), 7, 1),
                        Row(Vectors.dense(14.0), 8, 1),
                        Row(Vectors.dense(30.0), 9, 1),
                        Row(Vectors.dense(7.0), 10, 1))
    val drop3 = new Drop3()
    val label = instances.head.getInt(2)
    val unbalanced = false
    val selectInstances = drop3.returnIfOneClass(instances, unbalanced, label)
    assert(selectInstances.size == 7)
  }

  test("Solo hay solo una instancia, no se elimina ninguna muestra") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1))
    val drop3 = new Drop3()
    val instanceToRemove = drop3.drop3(instances, false, 4)
    assert(instanceToRemove == List())
  }

  test("Solo dos instancias") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1),
                        Row(Vectors.dense(66.0), 7, 1))
    val drop3 = new Drop3()
    val instanceToRemove = drop3.drop3(instances, false, 4)
    assert(instanceToRemove == List(3, 7))
  }
}
