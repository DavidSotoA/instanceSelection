package params

import org.apache.spark.sql.{DataFrame, SparkSession}

class IsParams (
  instances: DataFrame,
  unbalanced: Boolean,
  minorityClass: Int,
  spark: SparkSession,
  neighbors: Int,
  subBuckets: Int,
  distancesIntervale: Int) {

    def unpackParams(): (DataFrame, Boolean, Int, SparkSession, Int, Int, Int) = {
      (instances, unbalanced, minorityClass, spark, neighbors, subBuckets, distancesIntervale)
    }

  }
