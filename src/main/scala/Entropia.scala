package com.lsh

import scala.util.Random

import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._


object Entropia extends InstanceSelection {
  var totalInstances: Long = _

  override def instanceSelection(instances: DataFrame, unbalanced: Boolean ): DataFrame = {
    val aggEntropy = new AggEntropy()
    val entropyForSignature = instances.groupBy("signature")
                               .agg(aggEntropy(instances.col("label")).as("entropy"))

    val instancesWithEntropy = instances
          .join(entropyForSignature, instances("signature") === entropyForSignature("signature"))
          .drop(entropyForSignature("signature"))

    instancesWithEntropy.filter(x => pickInstance(x(4).asInstanceOf[Double]
                                     ,x(2).asInstanceOf[Int]))
  }

  def pickInstance(entropia: Double, label: Int): Boolean = {
    if(label == 1) {
      return true
    }

    val rnd = scala.util.Random.nextFloat
    if (rnd < entropia){
      return true
    }
    return false
  }

}

class AggEntropy() extends UserDefinedAggregateFunction {

 override def inputSchema: StructType = StructType(Array(StructField("item", IntegerType)))

 override def bufferSchema: StructType = StructType(Array(
   StructField("fraude", LongType),
   StructField("legal", LongType),
   StructField("total", LongType)
 ))

 override def dataType: DataType = DoubleType

 override def deterministic = true

 override def initialize(buffer: MutableAggregationBuffer) = {
   buffer(0) = 0L
   buffer(1) = 0L
   buffer(2) = 0L
 }

 override def update(buffer: MutableAggregationBuffer, input: Row) = {
   if (input.getInt(0) == 1) {
     buffer(0) = buffer.getLong(0) + 1
   } else {
     buffer(1) = buffer.getLong(1) + 1
   }
   buffer(2) = buffer.getLong(2) + 1
 }

 override def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
   buffer1(0) = buffer1.getLong(0) + buffer2.getLong(0)
   buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
   buffer1(2) = buffer1.getLong(2) + buffer2.getLong(2)
 }

 override def evaluate(buffer: Row) = {
   if(buffer.getLong(1).toDouble == buffer.getLong(2)){
     1.toDouble/buffer.getLong(2)
   } else {
     val numOfInstances = Array(buffer.getLong(0).toDouble , buffer.getLong(1).toDouble )
     Entropy.calculate(numOfInstances, buffer.getLong(2))
   }
 }
}
