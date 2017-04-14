package com.lsh

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
    val groupBySiganture = instances.groupBy(Constants.SET_OUPUT_COL_LSH).count
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
    title: String,
    info_LSH: (String, Int, Int, Double, Long, Long, Long, Double),
    info_instance_selection: (String, Double, Double)) {
      val (metodoLsh, ands, ors, timeLsh, numeroDeCubetas, maxValue, minValue, avgValue) = info_LSH
      val (metodoInstanceSelection, timeInstanceSelection, reduction) = info_instance_selection
      val strToWrite =
      title.toUpperCase +
      "\n\nLSH" +
      "\nmétodo: " + metodoLsh +
      "\nnúmero de ands: " + ands +
      "\nnúmero de ors: " + ors +
      "\ntiempo de ejecución: " + timeLsh +
      "\nnúmero de cubetas: " + numeroDeCubetas +
      "\ncubeta con más muestras: " + maxValue +
      "\ncubeta con menos muestras: " + minValue +
      "\npromedio de muestras por cubeta: " + avgValue +
      "\n\nINSTANCE SELECTION" +
      "\nmétodo: " + metodoInstanceSelection +
      "\ntiempo de ejecución: " + timeInstanceSelection +
      "\ncantidad de legales:" +
      "\ncantidad de fraudes:" +
      "\nreducción de datos: " + reduction +
      "\n\nTIEMPO TOTAL: " + (timeLsh +timeInstanceSelection) + "\n\n"
      if (Files.exists(Paths.get(fileToWrite))){
        Files.write(Paths.get(fileToWrite), strToWrite.getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND)
      } else {
        Files.write(Paths.get(fileToWrite), strToWrite.getBytes(StandardCharsets.UTF_8))
      }
    }
}
