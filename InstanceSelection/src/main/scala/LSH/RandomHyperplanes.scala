package lsh

import scala.util.Random

import utilities.Constants
import mathematics.Mathematics
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

case class RandomHyperplanes(
    dataset_RH: DataFrame,
    numHashTables_RH: Int,
    spark_RH: SparkSession) extends LSH {
  dataset = dataset_RH
  numHashTables = numHashTables_RH
  spark = spark_RH
  require(numHashTables > 0, "numHashTables debe ser mayor a cero")
  var hyperplanes = createHiperplanes()

  def getHyperplanes(): Array[Vector] = {
    hyperplanes
  }

  def setHyperplanes(set_hyperplanes: Array[Vector]) {
      hyperplanes = set_hyperplanes
  }

  def createHiperplanes(): Array[Vector] = {
    val inputDim = dataset.select(Constants.COL_FEATURES).head.get(0)
      .asInstanceOf[Vector].size
    Array.fill(numHashTables) {
      Vectors.dense(Array.fill(inputDim)(Random.nextDouble()))
    }
  }

  override def lsh(colForLsh: String): DataFrame = {
    val partiallyHashFunction = hashFunction( _ : Vector, hyperplanes)
    val transformUDF = udf(partiallyHashFunction)
    val signatureDF = dataset.withColumn(Constants.COL_SIGNATURE,
      transformUDF(dataset(colForLsh)))
    signatureDF.repartition(col(Constants.COL_SIGNATURE ))
  }

  override def hashFunction(instance: Vector,
    hashFunctions: Array[Vector]): String = {
     val signature = (dotRestult: Double) => {
       if (dotRestult >= 0) {
         1
       } else {
         0
       }
     }
     val binSignature = hashFunctions.map(hash => signature(Mathematics.dot(hash, instance)))
     Mathematics.stringSignature(binSignature)
  }

}
