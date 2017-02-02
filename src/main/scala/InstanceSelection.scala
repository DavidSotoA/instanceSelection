package com.lsh

import org.apache.spark.sql._

trait InstanceSelection {
  def instanceSelection(instances: Dataset[_], unbalanced: Boolean ): Dataset[_]
}
