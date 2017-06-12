package instanceSelection

import scala.util.Random

import utilities.Constants
import params.IsParams
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

object Entropia extends InstanceSelection {
  var totalInstances: Long = _

  override def instanceSelection(params: IsParams): DataFrame = {
    val (instances, unbalanced, minorityClass, spark, _, _ , _) = params.unpackParams()
    val aggEntropy = new AggEntropyUnbalanced()
    val sc = spark.sparkContext

    // se agrega la entropia a cada instacia, esta es calculada en base a las intancias de la cubeta
    val entropyForSignature = addEntropy(
                              instances,
                              (Constants.COL_SIGNATURE,
                               Constants.COL_LABEL,
                               Constants.COL_ENTROPY),
                               aggEntropy)

    sc.broadcast(entropyForSignature)

    //se hace join de de las entropias con los datos originales
    var selectInstances = instances.join(entropyForSignature, Constants.COL_SIGNATURE)

    // se hace la seleccion de las instances basada en la entropia obtenida
    selectInstances.filter(x => pickInstance(x(4).asInstanceOf[Double],
                                x(3).asInstanceOf[Int], unbalanced, minorityClass))
                    .drop(Constants.COL_SIGNATURE, Constants.COL_ENTROPY)
                    .dropDuplicates(Constants.COL_ID)
  }

  // metodo encargado de seleccionar las intancias en base a una entropia dada mediante la generacion de un numero aleatorio
  def pickInstance(entropia: Double, COL_LABEL: Int, unbalanced: Boolean, minorityClass: Int): Boolean = {
    if (COL_LABEL == minorityClass && unbalanced) {
      return true
    }
    val rnd = scala.util.Random.nextFloat
    if (rnd < entropia) {
      return true
    }
    return false
  }

  // metodo encargado de calcular la entropia por cubeta mediante la UDAF y asignarla a cada muestra
  def addEntropy(
    instances: DataFrame,
    columnNames: (String, String, String),
    aggEntropy: AggEntropyUnbalanced): DataFrame = {
      val (colSignature, colCOL_LABEL, colOutput) = columnNames
      instances
      .groupBy(colSignature)
      .agg(aggEntropy(instances.col(colCOL_LABEL))
      .as(colOutput))
  }
}

class AggEntropyUnbalanced() extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(Array(StructField("item", IntegerType)))

  override def bufferSchema: StructType =  StructType(Array(
    StructField("labels", MapType(IntegerType, LongType)),
    StructField("total", LongType)
  ))

  override def dataType: DataType = DoubleType

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Map[Int, Long]()
    buffer(1) = 0L
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val existLabel = buffer(0).asInstanceOf[Map[Int, Long]].keys.exists(_ == input.getInt(0))
    var newVal: Long = 1
    if (existLabel){
      newVal = buffer(0).asInstanceOf[Map[Int, Long]](input.getInt(0)) + 1
    }
    buffer(0) = buffer(0).asInstanceOf[Map[Int, Long]] + (input.getInt(0) -> newVal)
    buffer(1) = buffer.getLong(1) + 1
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1(0).asInstanceOf[Map[Int, Long]] ++ buffer2(0).asInstanceOf[Map[Int, Long]]
    buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
  }

  override def evaluate(buffer: Row): Any = {
    if (buffer(0).asInstanceOf[Map[Int, Long]].size == 1) {
      1.0/buffer.getLong(1)
    } else {
      val numOfInstances = buffer(0).asInstanceOf[Map[Int, Long]].map{case (a,b) => b.toDouble}.toArray
      Entropy.calculate(numOfInstances, buffer.getLong(1))
    }
  }
}
