package com.test

import instanceSelection.{AggEntropyUnbalanced, LSH_IS_F, Agg_LSH_Is_F_Unbalanced, Agg_LSH_Is_F_Balanced}
import utilities.{Constants, Utilities}
//import com.lsh.EntropiaLSH_IS_F
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class LSH_IS_F_Test extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _
  var sc : SparkContext = _

  override def beforeAll() {
    spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
    sc = spark.sqlContext.sparkContext
  }

  test("Se selecciona una instancia de cada clase por cubeta cuando las clases son desbalanceadas"){
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01"),
      (7, Vectors.dense(-0.4, -4.0), 1, "01"),
      (8, Vectors.dense(-0.5, -3.0), 1, "10"),
      (9, Vectors.dense(-0.5, -3.0), 1, "00"),
      (10, Vectors.dense(-0.5, -3.0), 1, "00"),
      (11, Vectors.dense(-0.5, -3.0), 1, "01"))
    ).toDF("idn", "features", "label", "signature")

    val aggIsF = new Agg_LSH_Is_F_Unbalanced()
    val isFDf = instances
                .groupBy("signature")
                .agg(aggIsF(instances("label"), instances("idn"))
                .as("select_instances"))
    val isF =  isFDf.collect

    assert(isF(0)(1).asInstanceOf[Seq[Int]] == List(7, 11, 6))
    assert(isF(1)(1).asInstanceOf[Seq[Int]] == List(3, 9, 10))
    assert(isF(2)(1).asInstanceOf[Seq[Int]] == List(5, 8))
  }

  test("Se selecciona una instancia de cada clase por cubeta cuando las clases son balanceadas"){
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00"),
      (9, Vectors.dense(-0.5, 3.0), -1, "00"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10"),
      (8, Vectors.dense(-0.5, -3.0), 1, "10"),
      (10, Vectors.dense(-0.5, -3.0), -1, "10"),
      (40, Vectors.dense(-0.5, -3.0), 5, "10"),
      (50, Vectors.dense(-0.5, -3.0), 5, "10"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01"),
      (11, Vectors.dense(-0.4, -4.0), -1, "01"),
      (12, Vectors.dense(-0.4, -4.0), -1, "01"),
      (13, Vectors.dense(-0.4, -4.0), -1, "01"),
      (14, Vectors.dense(-0.4, -4.0), -1, "01"),
      (15, Vectors.dense(-0.4, -4.0), -1, "01"),
      (16, Vectors.dense(-0.4, -4.0), 1, "01"),
      (17, Vectors.dense(-0.4, -4.0), 1, "11"),
      (18, Vectors.dense(-0.4, -4.0), 1, "11"),
      (19, Vectors.dense(-0.4, -4.0), 1, "11")
    )).toDF("idn", "features", "label", "signature")

    val aggIsF = new Agg_LSH_Is_F_Balanced()
    val isFDf = instances
                .groupBy("signature")
                .agg(aggIsF(instances("label"), instances("idn"))
                .as("select_instances"))
    val isF =  isFDf.select("select_instances").collect
    assert(isF(0)(0).asInstanceOf[Seq[Int]].size == 1)
    assert((isF(0)(0).asInstanceOf[Seq[Int]](0) == 19))

    assert(isF(1)(0).asInstanceOf[Seq[Int]].size == 1)
    assert((isF(1)(0).asInstanceOf[Seq[Int]](0) == 15))

    assert(isF(2)(0).asInstanceOf[Seq[Int]].size == 1)
    assert((isF(2)(0).asInstanceOf[Seq[Int]](0) == 9))

    assert(isF(3)(0).asInstanceOf[Seq[Int]].size == 3)
    assert((isF(3)(0).asInstanceOf[Seq[Int]](0) == 10))
    assert((isF(3)(0).asInstanceOf[Seq[Int]](1) == 8))
    assert((isF(3)(0).asInstanceOf[Seq[Int]](2) == 50))
  }
}
