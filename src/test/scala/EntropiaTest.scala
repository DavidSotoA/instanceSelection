package com.test

import com.lsh.AggEntropy
import com.lsh.Constants
import com.lsh.Entropia
import com.lsh.Utilities
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class EntropiaTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _
  var sc : SparkContext = _

  override def beforeAll() {
    spark = Utilities.initSparkSession
    sc = spark.sqlContext.sparkContext
  }

  test("Se calcula la entropia para cada clave") {
    val aggEntropy = new AggEntropy()
    val data = (1 to 1000).map{x: Int => x match {
    case t if (t >= 1 && t <= 300) => Row("A", 1)
    case t if (t > 300 && t <= 500) => Row("A", -1)
    case t if (t > 500 && t <= 600) => Row("B", 1)
    case t if (t > 600) => Row("B", -1)
    }}

    val schema = StructType(Array(
      StructField("key", StringType),
      StructField("label", IntegerType)
    ))

    val rdd = sc.parallelize(data)
    val df = spark.createDataFrame(rdd, schema)

    val entropyDF = df.groupBy("key").agg(aggEntropy(df.col("label")).as("entropy"))

    val entropyA = entropyDF.select("entropy").where("key == 'A'").head
    val entropyB = entropyDF.select("entropy").where("key == 'B'").head

    assert(entropyA(0).asInstanceOf[Double] == 0.9709505944546686)
    assert(entropyB(0).asInstanceOf[Double] == 0.7219280948873623)
  }

  test("Si solo hay muestras de la clase mayoritaria, la entropia sera el porcentaje de 1 muestra") {
    val aggEntropy = new AggEntropy()
    val data = (1 to 1000).map{x: Int => Row("A", -1)}

    val schema = StructType(Array(
      StructField("key", StringType),
      StructField("label", IntegerType)
    ))

    val rdd = sc.parallelize(data)
    val df = spark.createDataFrame(rdd, schema)

    val entropyDF = df.groupBy("key").agg(aggEntropy(df.col("label")).as("entropy"))

    val entropyA = entropyDF.select("entropy").where("key == 'A'").head

    assert(entropyA(0).asInstanceOf[Double] == 0.001)
  }
}
