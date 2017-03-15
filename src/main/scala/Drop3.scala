package com.lsh

import com.github.martincooper.datatable.{DataColumn, DataRow, DataTable, DataValue}

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction, Window}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

case class RowTable(id: (Int, Int),
                    distances: Seq[(Double, Int, Int)],
                    neighbors: Seq[(Double, Int, Int)],
                    enemy: Double,
                    associates: Seq[Int])

case class Drop3() {

  def instanceSelection(instances: DataFrame, unbalanced: Boolean, k_Neighbors: Int): DataFrame = {
    require((k_Neighbors%2 ==1) && (k_Neighbors > 0),
             "El numero de vecinos debe ser impar y positivo")
    val aggKnn = new AggKnn()
    val instancesWithInfo = instances.groupBy("signature").agg(aggKnn(instances.col("features"),
                                      instances.col("idn"), instances.col("label")).as("info"))

    val transformUDF = udf(drop3(_ : Seq[Row], k_Neighbors))
    instancesWithInfo.withColumn("pruebaCol", transformUDF(instancesWithInfo.col("info")))
  }

  def drop3(instances: Seq[Row], k_Neighbors: Int): Vector = {
    var table = completeTable(instances, k_Neighbors, createDataTable(instances))
    throw new IllegalArgumentException ("unimplement method")
  }

  def createDataTable(instances: Seq[Row]): DataTable = {
    val id_col = new DataColumn[(Int, Int)]("id_col", Array[(Int, Int)]())
    val dist_col = new DataColumn[Seq[(Double, Int, Int)]]("Distancias",
                                  Array[Seq[(Double, Int, Int)]]())
    val neighbors_col = new DataColumn[Seq[(Double, Int, Int)]]("Vecinos",
                                       Array[Seq[(Double, Int, Int)]]())
    val enemy_col = new DataColumn[Double]("Enemigo", Array[Double]())
    val associates_col = new DataColumn[Seq[Int]]("Asociados", Array[Seq[Int]]())
    var table = DataTable("NewTable", Seq(id_col, dist_col, neighbors_col, enemy_col,
                                          associates_col))

    val dist_null = DataValue.apply(null.asInstanceOf[Seq[(Double, Int, Int)]])
    val neighbors_null = dist_null
    val enemy_null = DataValue.apply(null.asInstanceOf[Double])
    val associates_null = DataValue.apply(Seq.empty[Int])

    val instanceSize = instances.size
    for(i <- 0 to (instanceSize-1)) {
      val id_tuple = (instances(i)(1).asInstanceOf[Int], instances(i)(2).asInstanceOf[Int])
      val value_id_tuple = DataValue.apply(id_tuple)
      val value_null = DataValue.apply(null)
      table = table.get.rows.add(value_id_tuple, dist_null, neighbors_null, enemy_null,
                                 associates_null)
    }
    table.get
  }

  def completeTable(instances: Seq[Row], k_Neighbors: Int, table: DataTable): DataTable = {
    var myTable = table
    var currentInstance: (Vector, Int, Int) = null.asInstanceOf[(Vector, Int, Int)]
    val instanceSize = instances.size
    for(i <- 0 to (instanceSize-1)) {
      currentInstance = (instances(i)(0).asInstanceOf[Vector],
                         instances(i)(1).asInstanceOf[Int],
                         instances(i)(2).asInstanceOf[Int])
      val instancesId = (currentInstance._2, currentInstance._3)
      val distancesOfCurrentInstance = calculateDistances(currentInstance._1, instances)
      val myNeighbors = findNeighbors(distancesOfCurrentInstance, k_Neighbors, false)
      val myEnemy = findMyNemesis(distancesOfCurrentInstance, currentInstance._3, false)

      // Traer indice y row de la instancia
      val (index, row) = getIndexAndRowById(instancesId._1, myTable)
      myTable = myTable.rows.replace(index, DataValue.apply(instancesId),
                                            DataValue.apply(distancesOfCurrentInstance),
                                            DataValue.apply(myNeighbors),
                                            DataValue.apply(myEnemy),
                                            DataValue.apply(row.associates)).get
      for(neighbor <- myNeighbors) {
        myTable = updateAssociates(instancesId._1, neighbor._2, myTable)
      }
    }
    myTable
  }

  def updateAssociates(associate: Int, instanceForUpdate: Int, table: DataTable): DataTable = {
    var myTable = table
    val (index, row) = getIndexAndRowById(instanceForUpdate, myTable)
    myTable.rows.replace(index, DataValue.apply(row.id),
                                DataValue.apply(row.distances),
                                DataValue.apply(row.neighbors),
                                DataValue.apply(row.enemy),
                                DataValue.apply(row.associates :+ associate)).get
  }

  def getIndexAndRowById(id: Int, table: DataTable): (Int, RowTable) = {
   val index = table.indexWhere(x => x.values(0).asInstanceOf[(Int, Int)]._1 == id)
   val row = table.filter(x => x.values(0).asInstanceOf[(Int, Int)]._1 == id).toDataTable(0)
   val rowT = new RowTable(row(0).asInstanceOf[(Int, Int)],
                           row(1).asInstanceOf[Seq[(Double, Int, Int)]],
                           row(2).asInstanceOf[Seq[(Double, Int, Int)]],
                           row(3).asInstanceOf[Double],
                           row(4).asInstanceOf[Seq[Int]])
   (index, rowT)
 }

  def findNeighbors(instances: Seq[(Double, Int, Int)],
                    k_Neighbors: Int,
                    needOrder: Boolean): Seq[(Double, Int, Int)] = {
    if (needOrder) {
      val instancesInOrder = scala.util.Sorting.stableSort(instances, (i1: (Double, Int, Int),
                                          i2: (Double, Int, Int)) => i1._1 < i2._1)
      return instancesInOrder.take(k_Neighbors + 1)
    }
    return instances.take(k_Neighbors + 1)
  }

  def findMyNemesis(instances: Seq[(Double, Int, Int)],
                    myLabel: Int,
                    needOrder: Boolean): Double = {
    val myEnemies = killFriends(instances, myLabel)
    if(needOrder) {
      var enemiesInOrder = scala.util.Sorting.stableSort(myEnemies, (i1: (Double, Int, Int),
                                          i2: (Double, Int, Int)) => i1._1 < i2._1)
      return enemiesInOrder.head._1
    }
    return myEnemies.head._1
  }

  def killFriends(instances: Seq[(Double, Int, Int)], myLabel: Int): Seq[(Double, Int, Int)] = {
    instances.filter(x => (x._3 != myLabel))
  }

  def calculateDistances(sample: Vector, instances: Seq[Row]): Seq[(Double, Int, Int)] = {
    val instanceSize = instances.size
    var distances = Array[(Double, Int, Int)]()
    for(i <- 0 to (instanceSize-1)) {
      val distance = Mathematics.distance(sample, instances(i)(0).asInstanceOf[Vector])
      if(distance != 0) {
        distances = distances :+ (distance, instances(i)(1).asInstanceOf[Int],
                                  instances(i)(2).asInstanceOf[Int])
      }
    }
    if(!distances.isEmpty) {
      distances = scala.util.Sorting.stableSort(distances, (i1: (Double, Int, Int),
                                            i2: (Double, Int, Int)) => i1._1 < i2._1)
    }
    distances
  }
}

class AggKnn() extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = StructType(Array(
    StructField("features", VectorType),
    StructField("idn", IntegerType),
    StructField("label", IntegerType)
  ))

   override def bufferSchema: StructType = StructType(Array(
     StructField("allInfo", ArrayType(StructType(Array(
                                       StructField("features", VectorType),
                                       StructField("idn", IntegerType),
                                       StructField("label", IntegerType)
                                     ))))
   ))

   override def dataType: DataType = ArrayType(StructType(Array(
                                     StructField("features", VectorType),
                                     StructField("idn", IntegerType),
                                     StructField("label", IntegerType)
                                    )))

   override def deterministic: Boolean = true

   override def initialize(buffer: MutableAggregationBuffer): Unit = {
     buffer(0) = Array[(Vector, Int, Int)]()
   }

   override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
     buffer(0) = buffer(0).asInstanceOf[Seq[(Vector, Int, Int)]] :+
        (input(0).asInstanceOf[Vector], input(1).asInstanceOf[Int], input(2).asInstanceOf[Int])
   }

   override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
     buffer1(0) = buffer1(0).asInstanceOf[Seq[(Vector, Int, Int)]] ++
                  buffer2(0).asInstanceOf[Seq[(Vector, Int, Int)]]
   }

   override def evaluate(buffer: Row): Any = {
     buffer(0)
   }

 }
