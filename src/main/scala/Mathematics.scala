package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.sql.DataFrame

object Mathematics{
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

  def stringSignature(numBin: Array[Int]): String = {
    var stringSignature = ""
    for(i <- 0 to (numBin.length-1)) {
      stringSignature = stringSignature + numBin(i).toString
    }
    stringSignature
  }

  def distance(a: Vector, b: Vector): Double = {
    Math.sqrt(Vectors.sqdist(a, b))
  }

  def normalizeVector(a : Vector): Vector ={
    val norm = Vectors.norm(a, 2)
    val values = a.toArray
    Vectors.dense(values.map(_/norm))
  }

  def normalize(df: DataFrame, inputCol: String): DataFrame = {
     val scaler = (new StandardScaler()
     .setInputCol(inputCol)
     .setOutputCol(Constants.SET_OUPUT_COL_SCALED)
     .setWithStd(true)
     .setWithMean(true))
     // Compute summary statistics by fitting the StandardScaler.
     val scalerModel = scaler.fit(df)
     val scaledData = scalerModel.transform(df)
     scaledData
   }
}
