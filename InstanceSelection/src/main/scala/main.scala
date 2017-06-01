package main

import instanceSelection.InstanceSelection
import reports.Report
import lsh.Lsh
import utilities.{Constants, Utilities, LogHelper}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._

object Main extends LogHelper {
  def main(args: Array[String]): Unit = {
    val spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_CLUSTER)
    val sc = spark.sqlContext.sparkContext

    logger.info("..........Leyendo parametros...............")
    val andFunctions = args(0).toInt
    val orFunctions = args(1).toInt
    val lshMethod = args(2)
    val instanceSelectionMethod = args(3)
    val base = spark.read.load(args(4))
    val urlReports = args(5)
    var sizeBucket = 0.0
    if (args.size > 6) {
      sizeBucket = args(6).toDouble
    }

    logger.info("..........Procesando datos...............")
    base.createOrReplaceTempView("test")
    val instances = spark.sql("SELECT " + Constants.COLS +
      " FROM test WHERE label!=0 and resp_code=1 or resp_code=2")
    val names = instances.columns
    val ignore = Array("idn", "label", "resp_code", "fraude", "nolabel")
    val selectFeatures = for (i <- names if !(ignore contains i )) yield i
    val vectorizedDF = Utilities.createVectorDataframe(selectFeatures, instances)

    val urlBase = "instanceSelection_" + lshMethod + "_con_" + instanceSelectionMethod + "_" + andFunctions + "_" + orFunctions + "_" + sizeBucket
    val urlLsh = urlBase +  "/lsh_" + lshMethod
    val urlInstanceSelection = urlBase + "/instanceSelection_" + instanceSelectionMethod

    logger.info("..........Ejecutando LSH...............")
    val instancesWithSignature = Lsh.lsh(lshMethod, vectorizedDF, spark, sizeBucket, andFunctions, orFunctions)
    val lshTime = Report.saveDFWithTime(instancesWithSignature, urlLsh, Constants.FORMAT_PARQUET)
    val signatureDF = spark.read.load(urlLsh)

    logger.info("..........Ejecutando instance selection...............")
    val instanceSelectionDF = InstanceSelection.instanceSelection(instanceSelectionMethod, signatureDF, spark, true)
    val instanceSelectionTime = Report.saveDFWithTime(instanceSelectionDF, urlInstanceSelection, Constants.FORMAT_PARQUET)
    val selectionDF = spark.read.load(urlInstanceSelection)

    logger.info("..........Realizando reportes...............")
    val (numeroDeCubetas, maxValue, minValue, avgValue) = Report.infoLSH(signatureDF)
    val reduction = Report.infoInstanceSelection(vectorizedDF, selectionDF)
    val lshInfo = (lshMethod, andFunctions, orFunctions, lshTime, numeroDeCubetas, maxValue, minValue, avgValue)
    val instanceSelectionInfo = (instanceSelectionMethod, instanceSelectionTime, reduction)
    Report.report(urlReports, lshInfo, instanceSelectionInfo)
  }
}
