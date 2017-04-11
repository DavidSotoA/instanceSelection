package com.lsh

import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

object Entropia extends InstanceSelection {
  var totalInstances: Long = _

  override def instanceSelection(instances: DataFrame, unbalanced: Boolean): DataFrame = {
    throw new IllegalArgumentException ("unimplement method")
  }

  def instanceSelection2(
    instances: DataFrame,
    unbalanced: Boolean,
    spark: SparkSession): DataFrame = {
      val aggEntropy = new AggEntropyUnbalanced()
      val sc = spark.sparkContext
      val entropyForSignature = addEntropy(
                                instances,
                                (Constants.SET_OUPUT_COL_LSH,
                                 Constants.LABEL,
                                 Constants.SET_OUPUT_COL_ENTROPY),
                                 aggEntropy)

      sc.broadcast(entropyForSignature)

      var selectInstances = instances.join(entropyForSignature, Constants.SET_OUPUT_COL_LSH)
      selectInstances.filter(x =>
                            pickInstance(x(4).asInstanceOf[Double], x(3).asInstanceOf[Int], true))
                            .drop(Constants.SET_OUPUT_COL_LSH, Constants.SET_OUPUT_COL_ENTROPY)

      selectInstances.dropDuplicates(Constants.INSTANCE_ID)
    }

  def pickInstance(entropia: Double, label: Int, unbalanced: Boolean): Boolean = {
    if (label == 1 && unbalanced) {
      return true
    }
    val rnd = scala.util.Random.nextFloat
    if (rnd < entropia) {
      return true
    }
    return false
  }

  def addEntropy(
    instances: DataFrame,
    columnNames: (String, String, String),
    aggEntropy: AggEntropyUnbalanced): DataFrame = {
      val (colSignature, colLabel, colOutput) = columnNames
      instances
      .groupBy(colSignature)
      .agg(aggEntropy(instances.col(colLabel))
      .as(colOutput))
  }
}

class AggEntropyUnbalanced() extends UserDefinedAggregateFunction {

 override def inputSchema: StructType = StructType(Array(StructField("item", IntegerType)))

 override def bufferSchema: StructType = StructType(Array(
   StructField("fraude", LongType),
   StructField("legal", LongType),
   StructField("total", LongType)
 ))

 override def dataType: DataType = DoubleType

 override def deterministic: Boolean = true

 override def initialize(buffer: MutableAggregationBuffer): Unit = {
   buffer(0) = 0L
   buffer(1) = 0L
   buffer(2) = 0L
 }

 override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
   if (input.getInt(0) == 1) {
     buffer(0) = buffer.getLong(0) + 1
   } else {
     buffer(1) = buffer.getLong(1) + 1
   }
   buffer(2) = buffer.getLong(2) + 1
 }

 override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
   buffer1(0) = buffer1.getLong(0) + buffer2.getLong(0)
   buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
   buffer1(2) = buffer1.getLong(2) + buffer2.getLong(2)
 }

 override def evaluate(buffer: Row): Any = {
   if (buffer.getLong(1).toDouble == buffer.getLong(2)) {
     1.0/buffer.getLong(2)
   } else {
     val numOfInstances = Array(buffer.getLong(0).toDouble, buffer.getLong(1).toDouble)
     Entropy.calculate(numOfInstances, buffer.getLong(2))
   }
 }

}
