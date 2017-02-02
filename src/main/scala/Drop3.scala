package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

case class Drop3(instances: Dataset[_]) extends InstanceSelection {

  override def instanceSelection(instances: Dataset[_], unbalanced: Boolean): Dataset[_] = {
    throw new IllegalArgumentException ("unimplement method")
  }
}

object Drop3 {

  def knn(instance: Vector, allInstances: DataFrame, numNeighbors: Int): Dataset[_] = {
    require(numNeighbors > 0, "El numero de vecinos debe ser mayor a 1")
    val partiallyDistanceFunction = Mathematics.distance(_ : Vector, instance)
    val keyDistUDF = udf(partiallyDistanceFunction)
    val allNeighbors = allInstances.withColumn("distance", keyDistUDF(allInstances("features")))
    allNeighbors.select("id", "distance").sort("distance").limit(numNeighbors)
  }

  def killFriends(instance: Row, allInstances: DataFrame): DataFrame = {
    val labelInstance = instance(2)
    allInstances.select("*").where("label != " + labelInstance)
  }
}
