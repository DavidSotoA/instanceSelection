package reports

import utilities.Constants
import org.apache.spark.sql.{DataFrame, SaveMode}
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets

object Report{

  def saveDFWithTime(instances: DataFrame, url: String, format: String): Double = {
    val t0 = System.nanoTime()
    instances.write.mode(SaveMode.Overwrite).format(format).save(url)
    val t1 = System.nanoTime()
    return (t1 - t0)/6e+10
  }

  def infoLSH(instances: DataFrame): (Long, Long, Long, Double) = {
    val groupBySiganture = instances.groupBy(Constants.COL_SIGNATURE).count
    val numeroDeCubetas = groupBySiganture.count
    val maxValue = groupBySiganture.groupBy().max("count").collect()(0)(0).asInstanceOf[Long]
    val minValue = groupBySiganture.groupBy().min("count").collect()(0)(0).asInstanceOf[Long]
    val avgValue = groupBySiganture.groupBy().avg("count").collect()(0)(0).asInstanceOf[Double]
    return (numeroDeCubetas, maxValue, minValue, avgValue)
  }

  def infoInstanceSelection(originalSet: DataFrame, selectedSet: DataFrame): Double = {
    val originalInstances = originalSet.count
    val selectedInstances = selectedSet.count
    return selectedInstances.toDouble/originalInstances
  }

  def report(
    fileToWrite: String,
    info_LSH: (String, Int, Int, Double, Long, Long, Long, Double),
    info_instance_selection: (String, Double, Double, Boolean)) {
      val (metodoLsh, ands, ors, timeLsh, numeroDeCubetas, maxValue, minValue, avgValue) = info_LSH
      val (metodoInstanceSelection, timeInstanceSelection, reduction, unbalanced) = info_instance_selection
      var strToWrite =
      metodoLsh + "," +
      metodoInstanceSelection + "," +
      unbalanced + "," +
      ands + "," +
      ors + "," +
      numeroDeCubetas + "," +
      maxValue + "," +
      minValue + "," +
      avgValue + "," +
      reduction + "," +
      timeLsh + "," +
      timeInstanceSelection + "," +
      (timeLsh + timeInstanceSelection)

      if (Files.exists(Paths.get(fileToWrite))){
        strToWrite = "\n" + strToWrite
        Files.write(Paths.get(fileToWrite), strToWrite.getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND)
      } else {
        strToWrite =
         "lsh_method,Is_method,unbalanced,ands,ors,bucket,max_bucket,min_bucket,avg_bucket,reduction,time_lsh,time_is,time_total\n" + strToWrite
        Files.write(Paths.get(fileToWrite), strToWrite.getBytes(StandardCharsets.UTF_8))
      }
    }
}
