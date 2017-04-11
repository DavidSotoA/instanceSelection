package com.test

import com.lsh.AggEntropyUnbalanced
import com.lsh.Constants
import com.lsh.Entropia
import com.lsh.Utilities
import com.lsh.Agg_LSH_Is_F
import com.lsh.LSH_IS_F
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

  test("Se selecciona una instancia de cada clase por cubeta"){
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00"),
      (9, Vectors.dense(-0.5, 3.0), -1, "00"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10"),
      (8, Vectors.dense(-0.5, -3.0), 1, "10"),
      (10, Vectors.dense(-0.5, -3.0), -1, "10"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01"),
      (7, Vectors.dense(-0.4, -4.0), 21, "01"))
    ).toDF("idn", "features", "label", "signature_1")

    val aggIsF = new Agg_LSH_Is_F()
    val isFDf =
      instances
      .groupBy("signature_1")
      .agg(aggIsF(instances("label"), instances("idn"))
      .as("select_instances"))
    val isF =  isFDf.select("select_instances").collect
    assert(isF(0)(0).asInstanceOf[Seq[Int]].size == 1)
    assert((isF(0)(0).asInstanceOf[Seq[Int]](0) == 6) || (isF(0)(0).asInstanceOf[Seq[Int]](0) == 2))

    assert(isF(1)(0).asInstanceOf[Seq[Int]].size == 1)
    assert((isF(1)(0).asInstanceOf[Seq[Int]](0) == 9) || (isF(1)(0).asInstanceOf[Seq[Int]](0) == 1))


    assert(isF(2)(0).asInstanceOf[Seq[Int]].size == 2)
    assert((isF(2)(0).asInstanceOf[Seq[Int]](0) == 10) || (isF(2)(0).asInstanceOf[Seq[Int]](0) == 4))
    assert((isF(2)(0).asInstanceOf[Seq[Int]](1) == 5) || (isF(2)(0).asInstanceOf[Seq[Int]](1) == 8))
  }
}
