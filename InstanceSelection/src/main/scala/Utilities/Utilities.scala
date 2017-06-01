package utilities

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

object Utilities {

  def initSparkContext(): SparkContext = {
    val conf = new SparkConf().setAppName(Constants.APP_NAME).setMaster(Constants.MASTER)
    new SparkContext(conf)
  }

  def initSparkSession(mode: String): SparkSession = {
    mode match {
      case Constants.SPARK_SESSION_MODE_CLUSTER => (
        SparkSession.builder()
        .appName(Constants.APP_NAME)
        .enableHiveSupport()
        .getOrCreate()
      )

      case Constants.SPARK_SESSION_MODE_LOCAL => (
        SparkSession.builder()
        .appName(Constants.APP_NAME)
        .master(Constants.MASTER)
        .config("spark.some.config.option", "some-value")
        .getOrCreate()
      )
    }
  }

  def createVectorDataframe(selectFeatures: Array[String], df: DataFrame): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(selectFeatures)
      .setOutputCol(Constants.COL_FEATURES)

    val vectorDF = assembler.transform(df)
    vectorDF.select(Constants.COL_ID, Constants.COL_FEATURES, Constants.COL_LABEL)
  }
}
