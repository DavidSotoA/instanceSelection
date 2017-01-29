package com.lsh

import scala.util.Random

import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql._

object Entropia extends InstanceSelection {
  var totalInstances: Long = _

  override def instanceSelection(instances: Dataset[_]): Dataset[_] = {
    val entropy = calcularEntropia(instances)
    if(entropy == 1.0) {
      return instances
    }
    if(entropy == 0.0) {
      return instances.limit(1)
    }
    return instances.sample(false, entropy, 1234)
  }

  def calcularEntropia(instances: Dataset[_]): Double = {
    val instancesC1 = instances.select("*").where(Constants.LABEL + "== 0").count
    val instancesC2 = instances.select("*").where(Constants.LABEL + "== 1").count
    val numOfInstances = Array(instancesC1.toDouble, instancesC2.toDouble)
    totalInstances = instancesC1 + instancesC2

    Entropy.calculate(numOfInstances, totalInstances)
  }
}
