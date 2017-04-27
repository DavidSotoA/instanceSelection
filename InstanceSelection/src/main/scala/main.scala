package com.lsh

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
    val instancesWithSignature = lsh(spark, sizeBucket, lshMethod, vectorizedDF, andFunctions, orFunctions)
    val lshTime = Report.saveDFWithTime(instancesWithSignature, urlLsh, Constants.FORMAT_PARQUET)
    val signatureDF = spark.read.load(urlLsh)

    logger.info("..........Ejecutando instance selection...............")
    val instanceSelectionDF = instanceSelection(signatureDF, spark, instanceSelectionMethod, true)
    val instanceSelectionTime = Report.saveDFWithTime(instanceSelectionDF, urlInstanceSelection, Constants.FORMAT_PARQUET)
    val selectionDF = spark.read.load(urlInstanceSelection)

    logger.info("..........Realizando reportes...............")
    val (numeroDeCubetas, maxValue, minValue, avgValue) = Report.infoLSH(signatureDF)
    val reduction = Report.infoInstanceSelection(vectorizedDF, selectionDF)
    val lshInfo = (lshMethod, andFunctions, orFunctions, lshTime, numeroDeCubetas, maxValue, minValue, avgValue)
    val instanceSelectionInfo = (instanceSelectionMethod, instanceSelectionTime, reduction)
    Report.report(urlReports, lshInfo, instanceSelectionInfo)
  }

  def lsh(
    spark: SparkSession,
    sizeBucket: Double,
    method: String,
    instances: DataFrame,
    andFunctions: Int,
    orFunctions: Int): DataFrame = {
      method match {
        case Constants.LSH_HYPERPLANES_METHOD => {
          val normalizeDF = Mathematics.normalize(instances, Constants.SET_OUPUT_COL_ASSEMBLER)
          val randomHyperplanes = new RandomHyperplanes(normalizeDF, andFunctions, spark)
          return randomHyperplanes.lsh(Constants.SET_OUPUT_COL_SCALED).drop(Constants.SET_OUPUT_COL_SCALED)
        }
        case Constants.LSH_PROJECTION_METHOD => {
          val normalizeDF = Mathematics.normalize(instances, Constants.SET_OUPUT_COL_ASSEMBLER)
          val randomProjection = new RandomProjectionLSH(normalizeDF, andFunctions, orFunctions, sizeBucket, spark)
          return randomProjection.lsh(Constants.SET_OUPUT_COL_SCALED).drop(Constants.SET_OUPUT_COL_SCALED)
        }
        case _ => throw new IllegalArgumentException("El método " + method + " no existe")
      }
    }

  def instanceSelection(
    instances: DataFrame,
    spark: SparkSession,
    method: String,
    unbalanced: Boolean): DataFrame = {
      method match {
        case Constants.INSTANCE_SELECTION_LSH_IS_S_METHOD => {
          return LSH_IS_S.instanceSelection(instances, unbalanced)
        }
        case Constants.INSTANCE_SELECTION_ENTROPY_METHOD => {
          return Entropia.instanceSelection2(instances, unbalanced, spark)
        }
        case Constants.INSTANCE_SELECTION_LSH_IS_F_METHOD => {
          return LSH_IS_F.instanceSelection(instances, unbalanced)
        }
        case _ => throw new IllegalArgumentException("El método " + method + " no existe")
      }
    }

}

 //Main.main2(Array("5","5","euclidean","lsh_is_s_serial","/user/augustosoto/test", "asda", "0.5" ))
 //serial_time = 0.9054728095833333
 //paralell_ time = 0.6538910134666667 