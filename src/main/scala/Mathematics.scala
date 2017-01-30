package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}

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

  def distance(a: Vector, b: Vector): Double = {
    Math.sqrt(Vectors.sqdist(a, b))
  }
}
