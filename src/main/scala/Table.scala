package com.lsh

import scala.collection.mutable.Stack

case class RowTable(id: (Int, Int),
                    distances: Seq[(Double, Int, Int)],
                    neighbors: Seq[(Double, Int, Int)],
                    enemy: Double,
                    associates: Seq[Int])

class Table(){
  var table = Stack[RowTable]()

  def addRow(row: RowTable) {
    table = table :+ row
  }

  def getTable(): Stack[RowTable] = {
    table
  }

  def getIndexAndRowById(id: Int): (Int, RowTable) = {
    val index = table.indexWhere(_.id._1 == id)
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
