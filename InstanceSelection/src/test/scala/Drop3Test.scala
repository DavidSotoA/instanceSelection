package com.test

import com.lsh.AggKnn
import com.lsh.Constants
import com.lsh.Drop3
import com.lsh.Mathematics
import com.lsh.RowTable
import com.lsh.Id
import com.lsh.Info
import com.lsh.Distances
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
    val distances = Drop3.calculateDistances(100, 0, instance, instances(0).asInstanceOf[Seq[Row]])
    assert(distances(0) == Info(1.0, 5, -1))
    assert(distances(1) == Info(10.63014581273465, 3, 1))
  }

  test("Se calculan las distancias en un intervalo dado") {
    val instances = Row(Seq(
      Row(Vectors.dense(9, 8), 3, 1),
      Row(Vectors.dense(1, 2), 5, -1),
      Row(Vectors.dense(12, 3), 8, 1),
      Row(Vectors.dense(5, 4), 2, -1)
    ))
    val instance = Vectors.dense(1, 1)
    val distances = Drop3.calculateDistances(100, 0, instance, instances(0).asInstanceOf[Seq[Row]])
    assert(distances(0) == Info(1.0, 5, -1))
    assert(distances(1) == Info(5.0,2,-1))
  }

  test("se recalculan las distancias") {
    val instances = Row(Seq(
      Row(Vectors.dense(9, 8), 3, 1),
      Row(Vectors.dense(1, 2), 5, -1),
      Row(Vectors.dense(12, 3), 8, 1),
      Row(Vectors.dense(5, 4), 2, -1)
    ))
  }

  test("Se encuentran los k vecinos mas cercanos en base a las distancias calculadas") {
    val instances = Seq( Info(1.0, 3, 1), Info(5.0, 3, 1),
                         Info(2.0, 3, 1), Info(20.0, 3, 1),
                         Info(456.0, 3, 1), Info(100.0, 3, 1))
    val instance = Vectors.dense(1, 1)
    val neighbors = Drop3.findNeighbors(instances, 3, true)
    assert(neighbors(0) == Info(1.0, 3, 1))
    assert(neighbors(1) == Info(2.0, 3, 1))
    assert(neighbors(2) == Info(5.0, 3, 1))
  }

  test("Se eliminan las instancias que tienen el mismo label") {
    val instances = Seq( Info(1.0, 3, 1), Info(5.0, 3, -1),
                         Info(2.0, 3, -1), Info(20.0, 3, 1),
                         Info(456.0, 3, 1), Info(100.0, 3, 1))
    val enemies = Drop3.killFriends(instances, 1)
    assert(enemies.size == 2)
  }

  test("Se encuentra al enemigo mas cercano") {
    val instances = Seq( Info(1.0, 3, 1), Info(5.0, 3, -1),
                         Info(2.0, 3, -1), Info(20.0, 3, 1),
                         Info(456.0, 3, 1), Info(100.0, 3, 1))
    val nemesis = Drop3.findMyNemesis(instances, 1, true)
    assert(nemesis == 2.0)
  }

  test("Se crea el datatable inicial") {
    val instances = Seq(Row(1.0, 3, 1), Row(5.0, 4, -1),
                        Row(2.0, 5, -1), Row(20.0, 6, 1))
    val table = Drop3.createDataTable(instances)
    assert(table.size == 4)
    assert(table.getRow(0).id == Id(3, 1))
    assert(table.getRow(1).id == Id(4, -1))
    assert(table.getRow(2).id == Id(5, -1))
    assert(table.getRow(3).id == Id(6, 1))
  }

  test("Dado un datatable y un id, se obtiene el indice y el row correpondiente a dicho id") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(20.0), 6, 1))
    val table = Drop3.createDataTable(instances)
    val (indice, row) = table.getIndexAndRowById(5)
    val waitedRow = new RowTable(Id(5, -1), null, null, 0.0, List())
    assert(indice == 2)
    assert(row.isInstanceOf[RowTable])
    assert(row == waitedRow)
  }

  test("Se actualizan las distancias") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                        Row(Vectors.dense(5.0), 4, -1),
                        Row(Vectors.dense(2.0), 5, -1),
                        Row(Vectors.dense(3.0), 6, -1),
                        Row(Vectors.dense(20.0), 7, 1),
                        Row(Vectors.dense(14.0), 8, 1),
                        Row(Vectors.dense(30.0), 9, -1),
                        Row(Vectors.dense(7.0), 10, 1))
    val delta = 5
    val table = Drop3.completeTable(instances, delta, 2, Drop3.createDataTable(instances))
    Drop3.recalculateDistances(9, delta, instances, table)

    assert(table.getRow(0).distances.info == Seq(Info(28.0,5,-1), Info(29.0,3,1)))

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
    val table = Drop3.completeTable(instances, 5, 3, Drop3.createDataTable(instances))

    val row1 = new RowTable(Id(7,1),
                  Distances(false, 5, Seq(Info(17.0,6,-1))),
                  Seq(Info(6.0,8,1), Info(10.0,9,-1), Info(13.0,10,1), Info(15.0,4,-1)),
                  10.0,
                  List(8, 9))

    val row0 = new RowTable(Id(9,-1),
                   Distances(false, 5, Seq(Info(27.0,6,-1))),
                   Seq(Info(10.0,7,1), Info(16.0,8,1), Info(23.0,10,1), Info(25.0,4,-1)),
                   10.0,
                   List(7))

    val row2 = new RowTable(Id(8,1),
                   Distances(false, 5, Seq(Info(12.0,5,-1))),
                   Seq(Info(6.0,7,1), Info(7.0,10,1), Info(9.0,4,-1), Info(11.0,6,-1)),
                   9.0,
                   List(7, 9))

    val row5 = new RowTable(Id(4,-1),
                   Distances(false, 5, Seq(Info(9.0,8,1))),
                   Seq(Info(2.0,6,-1), Info(2.0,10,1), Info(3.0,5,-1), Info(4.0,3,1)),
                   2.0,
                   List(3, 5, 6, 7, 8, 9, 10))

    val row4 = new RowTable(Id(6,-1),
                   Distances(false, 5, Seq(Info(11.0,8,1))),
                   Seq(Info(1.0,5,-1), Info(2.0,3,1), Info(2.0,4,-1), Info(4.0,10,1)),
                   2.0,
                   List(3, 4, 5, 8, 10))

    val row3 = new RowTable(Id(10,1),
                   Distances(false, 5, Seq(Info(7.0,8,1))),
                   Seq(Info(2.0,4,-1), Info(4.0,6,-1), Info(5.0,5,-1), Info(6.0,3,1)),
                   2.0,
                   List(3, 4, 5, 6, 7, 8, 9))

    val row7 = new RowTable(Id(3,1),
                   Distances(false, 5, Seq(Info(13.0,8,1))),
                   Seq(Info(1.0,5,-1), Info(2.0,6,-1), Info(4.0,4,-1), Info(6.0,10,1)),
                   1.0,
                   List(4, 5, 6, 10))


    val row6 = new RowTable(Id(5,-1),
                   Distances(false, 5, Seq(Info(12.0,8,1))),
                   Seq(Info(1.0,3,1), Info(1.0,6,-1), Info(3.0,4,-1), Info(5.0,10,1)),
                   1.0,
                   List(3, 4, 6, 10))

    assert(table.size == 8)
    assert(table.getRow(0) == row0)
    assert(table.getRow(1) == row1)
    assert(table.getRow(2) == row2)
    assert(table.getRow(3) == row3)
    assert(table.getRow(4) == row4)
    assert(table.getRow(5) == row5)
    assert(table.getRow(6) == row6)
    assert(table.getRow(7) == row7)
  }

  test("El metodo knn retorna la etiqueta que tiene la mayoria de los vecinos") {
    val labels = Seq(-1, 1)
    val neighbors = Seq(Info(2.2, 4, 1), Info(2.2, 4, 1), Info(2.2, 4, -1))
    val label = Drop3.knn(labels, neighbors)
    assert(label == 1)
  }

  test("Se determina si remover una instancia en base a la clasificacion de"
    + " los asociados de una instancia con p y sin p") {
      val instances = Seq(Row(Vectors.dense(1.0), 3, 1),
                          Row(Vectors.dense(5.0), 4, -1),
                          Row(Vectors.dense(2.0), 5, -1),
                          Row(Vectors.dense(3.0), 6, -1),
                          Row(Vectors.dense(20.0), 7, 1))
      val table = Drop3.completeTable(instances,100, 3, Drop3.createDataTable(instances))
      val remove = Drop3.removeInstance(Id(3, 1), List(4, 5, 6, 7), table, List())

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
    val table = Drop3.completeTable(instances, 5, 3, Drop3.createDataTable(instances))
    val updateTable = Drop3.updateTableForRemove(10, List(3, 4, 5, 6, 7, 8, 9), 5, instances, table, List())

    val row0 = new RowTable(Id(9,-1),
                   Distances(false,5,Seq()),
                   Seq(Info(10.0,7,1), Info(16.0,8,1), Info(25.0,4,-1), Info(27.0,6,-1)),
                   10.0,
                   List(7))

  val row1 = new RowTable(Id(7,1),
                   Distances(false,5,Seq()),
                   Seq(Info(6.0,8,1), Info(10.0,9,-1), Info(15.0,4,-1), Info(17.0,6,-1)),
                   10.0,
                   List(8, 9))

  val row2 = new RowTable(Id(8,1),
                  Distances(false,5,Seq()),
                  Seq(Info(6.0,7,1), Info(9.0,4,-1), Info(11.0,6,-1), Info(12.0,5,-1)),
                  9.0,
                  List(7, 9, 3, 4, 5, 6))

    val row3 = new RowTable(Id(4,-1),
                   Distances(false,5,Seq()),
                   Seq(Info(2.0,6,-1), Info(3.0,5,-1), Info(4.0,3,1), Info(9.0,8,1)),
                   2.0,
                   List(3, 5, 6, 7, 8, 9))

    val row4 = new RowTable(Id(6,-1),
                   Distances(false,5,Seq()),
                   Seq(Info(1.0,5,-1), Info(2.0,3,1), Info(2.0,4,-1), Info(11.0,8,1)),
                   2.0,
                   List(3, 4, 5, 8, 7, 9))

    val row5 = new RowTable(Id(10,1),
                   Distances(false, 5, Seq(Info(7.0,8,1))),
                   Seq(Info(2.0,4,-1), Info(4.0,6,-1), Info(5.0,5,-1), Info(6.0,3,1)),
                   2.0,
                   List(3, 4, 5, 6, 7, 8, 9))

    val row6 = new RowTable(Id(3,1),
                   Distances(false,5,Seq()),
                   Seq(Info(1.0,5,-1), Info(2.0,6,-1), Info(4.0,4,-1), Info(13.0,8,1)),
                   1.0,
                   List(4, 5, 6))

    val row7 = new RowTable(Id(5,-1),
                   Distances(false,5,Seq()),
                   Seq(Info(1.0,3,1), Info(1.0,6,-1), Info(3.0,4,-1), Info(12.0,8,1)),
                   1.0,
                   List(3, 4, 6, 8))

     assert(updateTable.size == 8)
     assert(updateTable.getRow(0) == row0)
     assert(updateTable.getRow(1) == row1)
     assert(updateTable.getRow(2) == row2)
     assert(updateTable.getRow(3) == row5)
     assert(updateTable.getRow(4) == row4)
     assert(updateTable.getRow(5) == row3)
     assert(updateTable.getRow(6) == row7)
     assert(updateTable.getRow(7) == row6)

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

    val instanceToRemove = Drop3.drop3(instances, 100, false, 3)
    assert(instanceToRemove == List(11, 12, 6, 7, 3, 4, 5))
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

    val instanceToRemove = Drop3.drop3(instances, 100, true, 4)
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

    val prueba = Drop3.instanceSelection(instances, true, 3, 5)
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
    val label = instances.head.getInt(2)
    val oneClasss = Drop3.isOneClass(instances, label)
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
    val label = instances.head.getInt(2)
    val oneClasss = Drop3.isOneClass(instances, label)
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
    val label = instances.head.getInt(2)
    val unbalanced = true
    val selectInstances = Drop3.returnIfOneClass(instances, unbalanced, label)
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
    val label = instances.head.getInt(2)
    val unbalanced = true
    val selectInstances = Drop3.returnIfOneClass(instances, unbalanced, label)
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
    val label = instances.head.getInt(2)
    val unbalanced = false
    val selectInstances = Drop3.returnIfOneClass(instances, unbalanced, label)
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
    val label = instances.head.getInt(2)
    val unbalanced = false
    val selectInstances = Drop3.returnIfOneClass(instances, unbalanced, label)
    assert(selectInstances.size == 7)
  }

  test("Solo hay solo una instancia, no se elimina ninguna muestra") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1))
    val instanceToRemove = Drop3.drop3(instances, 100, false, 4)
    assert(instanceToRemove == List())
  }

  test("Solo dos instancias") {
    val instances = Seq(Row(Vectors.dense(1.0), 3, -1),
                        Row(Vectors.dense(66.0), 7, 1))
    val instanceToRemove = Drop3.drop3(instances, 5, false, 2)
    assert(instanceToRemove == List(7, 3))
  }
}
