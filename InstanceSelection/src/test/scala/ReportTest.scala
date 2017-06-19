package com.test

import reports.Report
import utilities.{Utilities, Constants}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

class ReportTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll() {
    spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_LOCAL)
  }

  test("Se obtiene la información sobre el lsh a partir de un dataframe") {
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

   val (numeroDeCubetas, maxValue, minValue, avgValue) = Report.infoLSH(instances)
   assert(numeroDeCubetas == 3)
   assert(maxValue == 7)
   assert(minValue == 2)
   assert(avgValue == 3.6666666666666665)
  }
}
