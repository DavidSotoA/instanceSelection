package instanceSelection

import utilities.Constants
import params.IsParams
import org.apache.spark.sql.{DataFrame, SparkSession}

trait InstanceSelection {
  def instanceSelection(params: IsParams): DataFrame
}

object InstanceSelection {

  def minorityClass(df: DataFrame): Int = {
    val numOflabels = df.groupBy(Constants.COL_LABEL)
                        .count.sort("count")
                        .select(Constants.COL_LABEL).head
    numOflabels(0).asInstanceOf[Int]
  }

  def instanceSelection(
    method: String,
    instances: DataFrame,
    spark: SparkSession,
    unbalanced: Boolean,
    neighbors: Int = 0,
    subBuckets: Int = 1000,
    distancesIntervale: Int = 0): DataFrame = {
      var minority = 0
      if (unbalanced) {
        minority = minorityClass(instances)
      }
      val params = new IsParams(instances, unbalanced, minority, spark, neighbors, subBuckets, distancesIntervale)
      method match {
        case Constants.INSTANCE_SELECTION_LSH_IS_S_METHOD => {
          return LSH_IS_S.instanceSelection(params)
        }
        case Constants.INSTANCE_SELECTION_ENTROPY_METHOD => {
          return Entropia.instanceSelection(params)
        }
        case Constants.INSTANCE_SELECTION_LSH_IS_F_METHOD => {
          return LSH_IS_F.instanceSelection(params)
        }
        case Constants.INSTANCE_SELECTION_DROP3_METHOD => {
          return Drop3.instanceSelection(params)
        }
        case _ => throw new IllegalArgumentException("El método " + method + " no existe")
      }
    }

}
