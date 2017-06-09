package main

import scopt.OptionParser
import instanceSelection.InstanceSelection
import reports.Report
import lsh.Lsh
import utilities.{Constants, Utilities, LogHelper}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._

case class Config(andFunctions: Int = null.asInstanceOf[Int],
                 orFunctions: Int = null.asInstanceOf[Int],
                 lshMethod: String = null.asInstanceOf[String],
                 instanceSelectionMethod: String = null.asInstanceOf[String],
                 dataUrl: String = null.asInstanceOf[String],
                 reportsUrl: String = null.asInstanceOf[String],
                 sizeBucket: Int = null.asInstanceOf[Int],
                 neighbors: Int = null.asInstanceOf[Int],
                 maxBucketSize: Int = null.asInstanceOf[Int],
                 distancesIntervale: Int = null.asInstanceOf[Int])

object Main extends LogHelper with App {

  val params = new scopt.OptionParser[Config]("scopt") {
        head("scopt", "3.x")
        opt[Int]('a', "andFunctions") required() action { (x, c) =>
          c.copy(andFunctions = x) } text("number of and functions")
        opt[Int]('o', "orFunctions") required() action { (x, c) =>
          c.copy(orFunctions = x) } text("number of or functions")
        opt[String]('l', "lshMethod") required() action { (x, c) =>
          c.copy(lshMethod = x) } text("name of lsh method")
        opt[String]('i', "dataUrl") required() action { (x, c) =>
          c.copy(dataUrl = x) } text("path of the data")
        opt[String]('r', "reportsUrl") required() action { (x, c) =>
          c.copy(reportsUrl = x) } text("path of the reports")
        opt[Int]('s', "sizeBucket") required() action { (x, c) =>
          c.copy(sizeBucket = x) } text("bucket size for randomProjection")
        opt[Int]('n', "neighbors") required() action { (x, c) =>
          c.copy(neighbors = x) } text("number of neighbors")
        opt[Int]('m', "maxBucketSize") required() action { (x, c) =>
          c.copy(maxBucketSize = x) } text("max number of intances per bucket")
        opt[Int]('d', "distancesIntervale") required() action { (x, c) =>
          c.copy(distancesIntervale = x) } text("number of distances calculates in DROP3")
  }

  params.parse(args, Config()) match {
    case Some(config) => {
      val spark = Utilities.initSparkSession(Constants.SPARK_SESSION_MODE_CLUSTER)
      val sc = spark.sqlContext.sparkContext

      logger.info("..........Leyendo parametros...............")
      val andFunctions = config.andFunctions
      val orFunctions = config.orFunctions
      val lshMethod = config.lshMethod
      val instanceSelectionMethod = config.instanceSelectionMethod
      val vectorizedDF = spark.read.load(config.dataUrl)
      val urlReports = config.reportsUrl
      var sizeBucket = config.sizeBucket

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
    case None => println("Please use --help argument for usage")
  }
}
