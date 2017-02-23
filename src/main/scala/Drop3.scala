package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.expressions.Window


case class Drop3() extends InstanceSelection {

  override def instanceSelection(instances: DataFrame, unbalanced: Boolean): DataFrame = {
     val aggKnn = new AggKnn()
    // val ventana = Window.partitionBy(instances.col("signature"))
    // instances.withColumn("colOfDistances", aggKnn(instances.col("features")) over ventana)
    instances.groupBy("signature").agg(aggKnn(instances.col("features"), instances.col("idn"),
                      instances.col("label")).as("info"))
  }
}

object Drop3 {

  def knn(instance: Vector, allInstances: DataFrame, numNeighbors: Int): Dataset[_] = {
    require(numNeighbors > 0, "El numero de vecinos debe ser mayor a 1")
    val partiallyDistanceFunction = Mathematics.distance(_ : Vector, instance)
    val keyDistUDF = udf(partiallyDistanceFunction)
    val allNeighbors = allInstances.withColumn("distance", keyDistUDF(allInstances("features")))
    allNeighbors.select("id", "distance").sort("distance").limit(numNeighbors)
  }

  def killFriends(instance: Row, allInstances: DataFrame): DataFrame = {
    val labelInstance = instance(2)
    allInstances.select("*").where("label != " + labelInstance)
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
