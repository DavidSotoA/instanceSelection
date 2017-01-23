package com.lsh

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

object Utilities {

  def initSparkContext(): SparkContext = {
    val conf = new SparkConf().setAppName(Constants.APP_NAME).setMaster(Constants.MASTER)
    new SparkContext(conf)
  }

  def initSparkSession(): SparkSession = {
      SparkSession
    .builder()
    .appName(Constants.APP_NAME)
    .master(Constants.MASTER)
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
  }

  def dot(x: Vector, y: Vector): Double = {
    var k = 0
    var sum = 0.0
    while(k < x.size) {
      sum = sum + (x(k) * y(k))
      k = k + 1
    }
    sum
  }

  def binaryToDec(numBin: Array[Int]): Int = {
      var numDec = 0.0
      var exp = 0
      for(i <- (numBin.length-1) to 0 by -1) {
        numDec += scala.math.pow(2, exp) * numBin(i)
        exp += 1
      }
      numDec.toInt
  }

}
