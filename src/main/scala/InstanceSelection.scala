package com.lsh

import org.apache.spark.sql._

trait InstanceSelection {
  def instanceSelection(instances: DataFrame, unbalanced: Boolean): DataFrame
}
