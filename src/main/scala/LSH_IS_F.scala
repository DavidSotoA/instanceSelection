package com.lsh

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions.explode

object LSH_IS_F {

  def instanceSelection(
    instances: DataFrame,
    unbalanced: Boolean): DataFrame = {
      val aggLSH = new Agg_LSH_Is_F()
      var instancesSelected =
        instances
        .groupBy(Constants.SET_OUPUT_COL_LSH)
        .agg(aggLSH(instances.col(Constants.LABEL), instances.col(Constants.INSTANCE_ID))
        .as(Constants.PICK_INSTANCE))
        .drop(Constants.SET_OUPUT_COL_LSH)

      val explodeDF =
        instancesSelected
       .select(explode(instancesSelected(Constants.PICK_INSTANCE))
       .as(Constants.INSTANCE_ID)).distinct()

      instances
      .join(explodeDF, Constants.INSTANCE_ID)
      .dropDuplicates(Constants.INSTANCE_ID)
      .drop(Constants.SET_OUPUT_COL_LSH)
  }
}

class Agg_LSH_Is_F() extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(Array(
   StructField("label", IntegerType),
   StructField("id", IntegerType)
  ))

  override def bufferSchema: StructType =
    StructType(Array(
      StructField("pick_legal", IntegerType),
      StructField("pick_fraude", IntegerType),
      StructField("cant_legal", IntegerType),
      StructField("cant_fraude", IntegerType)
    ))

  override def dataType: DataType = ArrayType(IntegerType)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
   buffer(0) = -1
   buffer(1) = -1
   buffer(2) = 0
   buffer(3) = 0
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
     if ((input.getInt(0) == -1)) {
       buffer(0) = input.getInt(1)
       buffer(2) = buffer.getInt(2) + 1
     }

     if ((input.getInt(0) == 1)) {
       buffer(1) = input.getInt(1)
       buffer(3) = buffer.getInt(3) + 1
     }
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getInt(0)
    buffer1(1) = buffer1.getInt(1)
    if(buffer1.getInt(0) == -1){
      buffer1(0) = buffer2.getInt(0)
    }
    if(buffer1.getInt(1) == -1){
      buffer1(1) = buffer2.getInt(1)
    }
    buffer1(2) = buffer1.getInt(2) + buffer2.getInt(2)
    buffer1(3) = buffer1.getInt(3) + buffer2.getInt(3)
  }

  override def evaluate(buffer: Row): Any = {
    var instance_legal = buffer(0)
    var instance_fraude = buffer(1)
    if(buffer.getInt(2) == 1) {
      instance_legal = -1
    }
    if(buffer.getInt(3) == 1) {
      instance_fraude = -1
    }
   Array(instance_legal, instance_fraude).filter(_ != -1)
  }
}
