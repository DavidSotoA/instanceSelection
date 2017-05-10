package com.lsh

import scala.collection.mutable.Stack

case class Id(id:Int, label: Int)
case class Info(distance: Double, id: Int, label: Int)
case class Distances(isUpdate: Boolean, updateIndex: Int, info: Seq[Info])

case class RowTable(id: Id,
                    distances: Distances,
                    neighbors: Seq[Info],
                    enemy: Double,
                    associates: Seq[Int])

class Table(){
  var table = Seq[RowTable]()

  def addRow(row: RowTable) {
    table = table :+ row
  }

  def removeRow(i: Int) {
    table = table.patch(i, Nil, 1)
  }

  def size (): Int = {
    table.size
  }

  def getRow(i: Int): RowTable = {
    table(i)
  }

  def getTable(): Seq[RowTable] = {
    table
  }

  def getIndexAndRowById(id: Int): (Int, RowTable) = {
    val index = table.indexWhere(_.id.id == id)
    val row = table(index)
    (index, row)
  }

  def replaceRow(index: Int, row: RowTable) {
    table = table.updated(index, row)
  }

  def orderByEnemy() {
    table = table.sortBy(_.enemy).reverse
  }

}
