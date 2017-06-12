package instanceSelection

import utilities.Constants
import org.apache.spark.sql._

trait InstanceSelection {
  def instanceSelection(instances: DataFrame, unbalanced: Boolean): DataFrame
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
      method match {
        case Constants.INSTANCE_SELECTION_LSH_IS_S_METHOD => {
          return LSH_IS_S.instanceSelection(instances, unbalanced)
        }
        case Constants.INSTANCE_SELECTION_ENTROPY_METHOD => {
          return Entropia.instanceSelection2(instances, unbalanced, minority, spark)
        }
        case Constants.INSTANCE_SELECTION_LSH_IS_F_METHOD => {
          return LSH_IS_F.instanceSelection(instances, unbalanced)
        }
        case Constants.INSTANCE_SELECTION_DROP3_METHOD => {
          return Drop3.instanceSelection(instances, unbalanced, neighbors, subBuckets, distancesIntervale, spark)
        }
        case _ => throw new IllegalArgumentException("El método " + method + " no existe")
      }
    }

}
