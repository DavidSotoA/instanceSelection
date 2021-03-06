package com.test

import instanceSelection.AggEntropyUnbalanced
import utilities.{Constants, Utilities}
import instanceSelection.Entropia
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
    spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
    sc = spark.sqlContext.sparkContext
  }

  test("Se calcula la entropia para cada clave") {
    val aggEntropy = new AggEntropyUnbalanced()
    val data = (1 to 1500).map{x: Int => x match {
    case t if (t >= 1 && t <= 300) => Row("A", 3)
    case t if (t > 300 && t <= 500) => Row("A", -1)
    case t if (t > 500 && t <= 600) => Row("A", 2)

    case t if (t > 600 && t <= 1000) => Row("B", 3)
    case t if (t > 1000 && t <= 1200) => Row("B", -1)
    case t if (t > 1200 ) => Row("B", 2)
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

    assert(entropyA(0).asInstanceOf[Double] == 1.4591479170272448)
    assert(entropyB(0).asInstanceOf[Double] == 1.5304930567574826)
  }


  test("Si solo hay muestras de una la clase, la entropia sera el "
    + "porcentaje equivalente a 1 muestra") {
    val aggEntropy = new AggEntropyUnbalanced()
    val data = (1 to 1000).map{x: Int => x match {
    case t if (t >= 1 && t <= 300) => Row("A", 1)
    case t if (t > 300) => Row("B", -1)
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

    assert(entropyA(0).asInstanceOf[Double] == 0.0033333333333333335)
    assert(entropyB(0).asInstanceOf[Double] == 0.0014285714285714286)
  }

  test("El metodo de entropia selecciona las instancias"){
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01"),
      (7, Vectors.dense(-0.4, -4.0), -1, "01"),
      (8, Vectors.dense(-0.4, -4.0), -1, "01"),
      (9, Vectors.dense(-0.4, -4.0), -1, "01"),
      (10, Vectors.dense(-0.4, -4.0), -1, "01"),
      (11, Vectors.dense(-0.4, -4.0), 1, "01"))
    ).toDF("idn", "features", "label", "signature")
  }
/*
  test("Se halla la entropia de las instancias en el dataframe cuando las clases son desbalanceadas"){
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00", "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01", "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00", "10"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10", "00"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10", "01"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01", "10"),
      (7, Vectors.dense(-0.4, -4.0), 21, "01", "10"))
    ).toDF("idn", "features", "label", "signature_1", "signature_2")

    val aggEntropy = new AggEntropyUnbalanced()
    val entropyDF = Entropia.addEntropy(instances, ("signature_1", "label", "entropy_1"), aggEntropy)
    val entropy =  entropyDF.select("entropy_1").collect
    assert(entropy(0)(0).asInstanceOf[Double] == 0.5)
    assert(entropy(1)(0).asInstanceOf[Double] == 1.0)
    assert(entropy(2)(0).asInstanceOf[Double] == 1.0)
  }

  test("Se encuentra la entropia para varias familias de funciones, con clases desbalanceadas"){
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(3.0, 0.5), -1 , "00", "00"),
      (1, Vectors.dense(4.0, 0.4), -1, "01", "01"),
      (2, Vectors.dense(-0.5, 3.0), 1, "00", "10"),
      (3, Vectors.dense(-0.4, 4.0), -1, "10", "00"),
      (4, Vectors.dense(-0.5, -3.0), 1, "10", "01"),
      (5, Vectors.dense(-0.4, -4.0), -1, "01", "10"))
    ).toDF("idn", "features", "label", "signature_1", "signature_2")

    val entropyDF = Entropia.instanceSelection2(instances, true, 2, sc)
    assert(1 == 1)
  }

  test("Se seleccionan las instancias luego de aplicar la entropia a todas las familias,  mediate una union") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(3.0, 0.5), -1 , "1","0"),
      (1, Vectors.dense(4.0, 0.4), -1,  "0","1"),
      (2, Vectors.dense(-0.5, 3.0), 1,  "1","1"),
      (3, Vectors.dense(-0.4, 4.0), -1, "0", "0"),
      (4, Vectors.dense(-0.5, -3.0), 1, "1", "1"),
      (5, Vectors.dense(-0.4, -4.0), -1, "0", "1"))
    ).toDF("idn", "features", "label", "pick_instance_1", "pick_instance_2")

    val method = Constants.ENTROPY_OR_METHOD
    val pickStr = "pick_instance_1, pick_instance_2"
    val entropyDF =  Entropia.selectIntancesFromAllFamilies(instances, method, spark, pickStr)
    val entropy = entropyDF.collect
    assert(entropy(0)(3).asInstanceOf[Int] == 1)
    assert(entropy(1)(3).asInstanceOf[Int] == 1)
    assert(entropy(2)(3).asInstanceOf[Int] == 1)
    assert(entropy(3)(3).asInstanceOf[Int] == 0)
    assert(entropy(4)(3).asInstanceOf[Int] == 1)
    assert(entropy(5)(3).asInstanceOf[Int] == 1)
  }

  test("Se seleccionan las instancias luego de aplicar la entropia a todas las familias,  mediate una interseccion") {
    val instances = spark.createDataFrame(Seq(
      (0, Vectors.dense(3.0, 0.5), -1 , "1","0"),
      (1, Vectors.dense(4.0, 0.4), -1,  "0","1"),
      (2, Vectors.dense(-0.5, 3.0), 1,  "1","1"),
      (3, Vectors.dense(-0.4, 4.0), -1, "0", "0"),
      (4, Vectors.dense(-0.5, -3.0), 1, "1", "1"),
      (5, Vectors.dense(-0.4, -4.0), -1, "0", "1"))
    ).toDF("idn", "features", "label", "pick_instance_1", "pick_instance_2")

    val method = Constants.ENTROPY_AND_METHOD
    val pickStr = "pick_instance_1, pick_instance_2"
    val entropyDF =  Entropia.selectIntancesFromAllFamilies(instances, method, spark, pickStr)
    val entropy = entropyDF.collect
    assert(entropy(0)(3).asInstanceOf[Int] == 0)
    assert(entropy(1)(3).asInstanceOf[Int] == 0)
    assert(entropy(2)(3).asInstanceOf[Int] == 1)
    assert(entropy(3)(3).asInstanceOf[Int] == 0)
    assert(entropy(4)(3).asInstanceOf[Int] == 1)
    assert(entropy(5)(3).asInstanceOf[Int] == 0)
  }*/

}
