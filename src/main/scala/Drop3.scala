package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.expressions.Window
import com.github.martincooper.datatable.{DataColumn, DataTable, DataValue, DataRow}

case class Drop3(){

  def instanceSelection(instances: DataFrame, unbalanced: Boolean, k_Neighbors: Int): DataFrame = {
    require((k_Neighbors%2 ==1) && (k_Neighbors > 0), "El numero de vecinos debe ser impar y positivo")
     val aggKnn = new AggKnn()
    // val ventana = Window.partitionBy(instances.col("signature"))
    // instances.withColumn("colOfDistances", aggKnn(instances.col("features")) over ventana)
    val instancesWithInfo = instances.groupBy("signature").agg(aggKnn(instances.col("features"), instances.col("idn"),
                                      instances.col("label")).as("info"))

    val transformUDF = udf(drop3(_ : Seq[Row], k_Neighbors))
    instancesWithInfo.withColumn("pruebaCol", transformUDF(instancesWithInfo.col("info")))
  }

  def drop3(instances: Seq[Row], k_Neighbors: Int): Vector = {
    val instanceSize = instances.size
    var currentInstance: (Vector, Int, Int) = null.asInstanceOf[(Vector,Int, Int)]
    for(i <- 0 to (instanceSize-1)){
      currentInstance = (instances(i)(0).asInstanceOf[Vector], instances(i)(1).asInstanceOf[Int],
                         instances(i)(2).asInstanceOf[Int])
      val intancesId = (currentInstance._2, currentInstance._3)
      val distancesOfCurrentInstance = calculateDistances(currentInstance._1, instances)
      val myNeighbors = findNeighbors(distancesOfCurrentInstance, k_Neighbors, false)
      val myEnemy = findMyNemesis(distancesOfCurrentInstance, currentInstance._3, false)

    }
    throw new IllegalArgumentException ("unimplement method")
  }

  def createDataTable(instances: Seq[Row]): DataTable = {
    val id_col = new DataColumn[(Int, Int)]("id_col", Array[(Int, Int)]())
    val dist_col = new DataColumn[Array[(Double, Int, Int)]]("Distancias", Array[Array[(Double, Int, Int)]]())
    val vecinos_col = new DataColumn[Array[(Double, Int, Int)]]("Vecinos", Array[Array[(Double, Int, Int)]]())
    val enemy_col = new DataColumn[Double]("Enemigo", Array[Double]())

    var table = DataTable("NewTable", Seq(id_col,  dist_col, vecinos_col, enemy_col))
    for(i <- 0 to (instanceSize-1)){
      id_tuple = (instances(i)(1).asInstanceOf[Int], instances(i)(2).asInstanceOf[Int])
      val value_id_tuple = DataValue.apply(id_tuple)
      val value_null = DataValue.apply(null)
      table = table.get.rows.add(value_id_tuple, value_null, value_null, value_null, value_null)
    }
    table
  }

  def findNeighbors(instances: Seq[(Double, Int, Int)], k_Neighbors: Int, needOrder: Boolean): Seq[(Double, Int, Int)] = {
    if(needOrder){
      val instancesInOrder = scala.util.Sorting.stableSort(instances,(i1: (Double, Int, Int),
                                          i2: (Double, Int, Int)) => i1._1 < i2._1)
      return instancesInOrder.take(k_Neighbors + 1)
    }
    return instances.take(k_Neighbors + 1)
  }

  def findMyNemesis(instances: Seq[(Double, Int, Int)], myLabel: Int, needOrder: Boolean): Double = {
    val myEnemies = killFriends(instances, myLabel)
    if(needOrder){
      var enemiesInOrder = scala.util.Sorting.stableSort(myEnemies,(i1: (Double, Int, Int),
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
    for(i <- 0 to (instanceSize-1)){
      val distance = Mathematics.distance(sample, instances(i)(0).asInstanceOf[Vector])
      distances = distances :+ (distance, instances(i)(1).asInstanceOf[Int],
                                instances(i)(2).asInstanceOf[Int])
    }
    if(!distances.isEmpty){
      distances = scala.util.Sorting.stableSort(distances,(i1: (Double, Int, Int),
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

   override def deterministic = true

   override def initialize(buffer: MutableAggregationBuffer) = {
     buffer(0) = Array[(Vector, Int, Int)]()
   }

   override def update(buffer: MutableAggregationBuffer, input: Row) = {
     buffer(0) = buffer(0).asInstanceOf[Seq[(Vector, Int, Int)]] :+
        (input(0).asInstanceOf[Vector], input(1).asInstanceOf[Int], input(2).asInstanceOf[Int])
   }

   override def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
     buffer1(0) = buffer1(0).asInstanceOf[Seq[(Vector, Int, Int)]] ++
                  buffer2(0).asInstanceOf[Seq[(Vector, Int, Int)]]
   }

   override def evaluate(buffer: Row) = {
     buffer(0)
   }

 }
