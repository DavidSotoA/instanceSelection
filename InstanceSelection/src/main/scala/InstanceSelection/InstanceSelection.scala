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

  /** método para realizar instance selection sobre un conjunto de instancias
   *  @param method: indica el método de instance selection a utilizar, sus poibles valores son:
   *  "entropia", "drop3", "lsh_is_s", "lsh_is_f"
   *  @param instances: conjunto de instancias a las que se les aplicara el instance selection
   *  @param spark: se autoexplica
   *  @param unbalanced: toma los valores de verdadero o false para indicar si el conjunto de instancias
   * es desbalanceado o no respectivamente
   *  @param neighbors: indica el numero de vecinos que se usara en el drop3
   *  @param subBuckets: indica el número maximo de muestras por cubeta en el drop3 (1000 por defecto)
   *  @param distancesIntervale: esto hay que quitarlo(poner 100)
   *  @return retorna el dataframe ingresado en el parametro instances luego de aplicarle instance selection
  */
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
