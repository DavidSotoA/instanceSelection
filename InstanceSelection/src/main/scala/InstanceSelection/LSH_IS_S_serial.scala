package instanceSelection

import utilities.Constants
import lsh.RandomProjectionLSH
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions.explode
import instanceSelection.Agg_LSH_Is_S_Balanced
import instanceSelection.Agg_LSH_Is_S_Unbalanced

object LSH_IS_S_serial {

  def instanceSelection(
    instances: DataFrame,
    unbalanced: Boolean,
    ors: Int,
    ands: Int,
    sizeBucket: Double,
    rowForLsh: String,
    spark: SparkSession): DataFrame = {
      val ors_lsh_is_s_serial = 1
      var aggLSH: UserDefinedAggregateFunction = new Agg_LSH_Is_S_Balanced()
      if (unbalanced) {
        aggLSH = new Agg_LSH_Is_S_Unbalanced()
      }

      val randomProjection = new RandomProjectionLSH(instances, ands, ors_lsh_is_s_serial , sizeBucket, spark)
      var instanceSelectionByOr = randomProjection.lsh(rowForLsh)

      instanceSelectionByOr =
        instanceSelectionByOr.groupBy(Constants.COL_SIGNATURE).agg(aggLSH(instanceSelectionByOr.col(Constants.COL_LABEL), instanceSelectionByOr.col(Constants.COL_ID)).as(Constants.PICK_INSTANCE)).drop(Constants.COL_SIGNATURE)

      instanceSelectionByOr =instanceSelectionByOr.select(explode(instanceSelectionByOr(Constants.PICK_INSTANCE)).as(Constants.COL_ID)) //distinct() ?

       var instanceSelection = instanceSelectionByOr

       for (i <- 1 until ors) {
         val randomProjection = new RandomProjectionLSH(instances, ands, ors_lsh_is_s_serial , sizeBucket, spark)
         var instanceSelectionByOr_i = randomProjection.lsh(rowForLsh)

         instanceSelectionByOr_i =
           instanceSelectionByOr_i.groupBy(Constants.COL_SIGNATURE).agg(aggLSH(instanceSelectionByOr_i.col(Constants.COL_LABEL), instanceSelectionByOr_i.col(Constants.COL_ID)).as(Constants.PICK_INSTANCE)).drop(Constants.COL_SIGNATURE)

         instanceSelectionByOr_i =instanceSelectionByOr_i.select(explode(instanceSelectionByOr_i(Constants.PICK_INSTANCE)).as(Constants.COL_ID)) //distinct() ?

        instanceSelection = instanceSelection.union(instanceSelectionByOr_i)
       }

      instances
      .join(instanceSelection, Constants.COL_ID)
      .dropDuplicates(Constants.COL_ID)
      .drop(Constants.COL_SIGNATURE)
  }
}
