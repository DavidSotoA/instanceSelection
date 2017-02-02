package com.lsh

import scala.util.Random

import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql._

object Entropia extends InstanceSelection {
  var totalInstances: Long = _
  var instancesMajorityClass: Dataset[_] = _

  override def instanceSelection(instances: Dataset[_], unbalanced: Boolean ): Dataset[_] = {
    instancesMajorityClass = instances.select("*").where(Constants.LABEL + "== -1")
    val instancesC1 = instancesMajorityClass.count
    val instancesC2 = instances.select("*").where(Constants.LABEL + "== 1").count

    if(unbalanced && instancesC1 == 0){
      return instances
    }

    if((unbalanced && instancesC2 == 0) || (!unbalanced && (instancesC1 == 0 || instancesC2 == 0))){
      return instances.limit(1)
    }

    val numOfInstances = Array(instancesC1.toDouble, instancesC2.toDouble)
    totalInstances = instancesC1 + instancesC2
    val entropy = Entropy.calculate(numOfInstances, totalInstances)

    if(unbalanced) {
      return instancesMajorityClass.sample(false, entropy, 1234)
    }else{
      return instances.sample(false, entropy, 1234)
    }
  }

  def calcularEntropia(instances: Dataset[_]): Double = {
    val instancesC1 = instances.select("*").where(Constants.LABEL + "== 1").count
    val instancesC2 = instances.select("*").where(Constants.LABEL + "== -1").count
    val numOfInstances = Array(instancesC1.toDouble, instancesC2.toDouble)
    totalInstances = instancesC1 + instancesC2

    Entropy.calculate(numOfInstances, totalInstances)
  }
}
