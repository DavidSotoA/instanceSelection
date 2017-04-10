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
    orsFunctions: Int,
    spark: SparkSession,
    method: String): DataFrame = {
      val aggEntropy = new AggEntropyUnbalanced()
      val partiallypickInstance = pickInstance( _ : Double, _ : Int, unbalanced)
      val transformUDF = udf(partiallypickInstance)
      var selectInstances = instances
      var pickStr = ""
      val sc = spark.sparkContext
      for (i <- 1 to orsFunctions) {
        val entropyForSignature = addEntropy(
                                  instances,
                                  (Constants.SET_OUPUT_COL_LSH + "_" + i,
                                  Constants.LABEL,
                                  Constants.SET_OUPUT_COL_ENTROPY + "_" + i),
                                  aggEntropy)

        sc.broadcast(entropyForSignature)

        selectInstances = selectInstances
                          .join(entropyForSignature, Constants.SET_OUPUT_COL_LSH + "_" + i)

        selectInstances = selectInstances
                         .withColumn((Constants.PICK_INSTANCE + "_" + i),
                          transformUDF(selectInstances(Constants.SET_OUPUT_COL_ENTROPY + "_" + i),
                                       selectInstances(Constants.LABEL)))
                          .drop(Constants.SET_OUPUT_COL_ENTROPY + "_" + i,
                                Constants.SET_OUPUT_COL_LSH + "_" + i)

        pickStr = pickStr + Constants.PICK_INSTANCE + "_" + i
        if(i < orsFunctions){
          pickStr = pickStr + " ,"
        }

      }
      val checkInstances = selectIntancesFromAllFamilies(selectInstances, method, spark, pickStr)
      checkInstances.filter(x => x(3).asInstanceOf[Int] == 1).drop(Constants.PICK_INSTANCE)
  }

  def selectIntancesFromAllFamilies(
    df: DataFrame,
    method: String,
    spark: SparkSession,
    pickStr: String): DataFrame = {
      df.registerTempTable("df")
      val concatDF = spark.sql("select idn, features, label, concat("+ pickStr +") as "
                              + Constants.PICK_CONCAT + " from df")

      var transformUDF = udf(entropyByAnd(_ : String))
      if(method ==  Constants.ENTROPY_OR_METHOD) {
        transformUDF = udf(entropyByOr(_ : String))
      }
      concatDF.withColumn(Constants.PICK_INSTANCE, transformUDF(concatDF(Constants.PICK_CONCAT)))
              .drop(Constants.PICK_CONCAT)
  }

  def entropyByAnd(values: String): Int = {
    val valuesList = values.toList
    if(valuesList.forall(_ == '1')) {
      return 1
    }
    return 0
  }

  def entropyByOr(values: String): Int = {
    val valuesList = values.toList
    if (valuesList.exists(_ == '1')) {
      return 1
    }
    return 0
  }

  /* Este método es llamado para hallar la entropia dado un conjunto de cubetas*/
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

  def pickInstance(entropia: Double, label: Int, unbalanced: Boolean): Int = {
    if (label == 1 && unbalanced) {
      return 1
    }
    val rnd = scala.util.Random.nextFloat
    if (rnd < entropia) {
      return 1
    }
    return 0
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
