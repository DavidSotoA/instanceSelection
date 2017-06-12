package com.test

import instanceSelection.{LSH_IS_S, AggEntropyUnbalanced, Agg_LSH_Is_S_Unbalanced, Agg_LSH_Is_S_Balanced}
import utilities.{Constants, Utilities}
//import com.lsh.Entropia
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class LSH_IS_S_Test extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _
  var sc : SparkContext = _

  override def beforeAll() {
    spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
    sc = spark.sqlContext.sparkContext
  }

  test("Se selecciona una instancia de cada clase por cubeta con clases desbalanceadas"){
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

    val aggIsS = new Agg_LSH_Is_S_Unbalanced()
    val isSDf = instances.groupBy("signature").agg(aggIsS(instances("label"), instances("idn")).as("select_instances"))
    val isS =  isSDf.select("select_instances").collect
    assert(isS(0)(0).asInstanceOf[Seq[Int]] == List(7, 11, 6))
    assert(isS(1)(0).asInstanceOf[Seq[Int]] == List(3, 9, 10, 1))
    assert(isS(2)(0).asInstanceOf[Seq[Int]] == List(5, 8, 4))
  }

  test("Se selecciona una instancia de cada clase por cubeta con clases balanceadas"){
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01"),
      (7, Vectors.dense(-0.4, -4.0), 21, "01"))
    ).toDF("idn", "features", "label", "signature")

    val aggIsS = new Agg_LSH_Is_S_Balanced()
    val isSDf = instances
                .groupBy("signature")
                .agg(aggIsS(instances("label"), instances("idn"))
                .as("select_instances"))
    val isS =  isSDf.select("select_instances").collect
    assert(isS(0)(0).asInstanceOf[Seq[Int]].size == 1)
    assert( (isS(0)(0).asInstanceOf[Seq[Int]](0) == 2) || (isS(0)(0).asInstanceOf[Seq[Int]](0) == 6))

    assert(isS(1)(0).asInstanceOf[Seq[Int]].size == 2)
    assert( (isS(1)(0).asInstanceOf[Seq[Int]](0) == 1))
    assert( (isS(1)(0).asInstanceOf[Seq[Int]](1) == 3))

    assert(isS(2)(0).asInstanceOf[Seq[Int]].size == 2)
    assert( (isS(2)(0).asInstanceOf[Seq[Int]](0) == 4))
    assert( (isS(2)(0).asInstanceOf[Seq[Int]](1) == 5))
  }

  test("Se selecionan las instancias mediante LSH_IS_S") {
    val instances = spark.createDataFrame(Seq(
      (1, Vectors.dense(3.0, 0.5), -1 , "00"),
      (2, Vectors.dense(4.0, 0.4), -1, "01"),
      (3, Vectors.dense(-0.5, 3.0), 1, "00"),
      (4, Vectors.dense(-0.4, 4.0), -1, "10"),
      (5, Vectors.dense(-0.5, -3.0), 1, "10"),
      (6, Vectors.dense(-0.4, -4.0), -1, "01"),
      (7, Vectors.dense(-0.4, -4.0), 1, "01"))
    ).toDF("idn", "features", "label", "signature")

    val orsFunctions = 2
    val unbalanced = false
    val instancesSelectedDF = LSH_IS_S.instanceSelection(instances, unbalanced)
    val instancesSelected =instancesSelectedDF.collect

    assert(instancesSelected(0)(0).asInstanceOf[Int] == 1 )
    assert(instancesSelected(1)(0).asInstanceOf[Int] == 6 )
    assert(instancesSelected(2)(0).asInstanceOf[Int] == 3 )
    assert(instancesSelected(3)(0).asInstanceOf[Int] == 5 )
    assert(instancesSelected(4)(0).asInstanceOf[Int] == 4 )
    assert(instancesSelected(5)(0).asInstanceOf[Int] == 7 )
  }
}
