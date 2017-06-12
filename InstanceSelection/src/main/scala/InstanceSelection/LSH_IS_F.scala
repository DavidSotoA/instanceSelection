package instanceSelection

import utilities.Constants
import params.IsParams
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions.explode

object LSH_IS_F extends InstanceSelection {

  override def instanceSelection(params: IsParams): DataFrame = {
      val (instances, unbalanced, minorityClass, _, _, _ , _) = params.unpackParams()
      // se determina la UDAF a usar en base a si las clases son desbañanceadas o balanceadas
      var aggLSH: UserDefinedAggregateFunction = new Agg_LSH_Is_S_Balanced()
      if (unbalanced) {
        aggLSH = new Agg_LSH_Is_S_Unbalanced()
      }

      // se eligen las instancias mediante una UDAF
      var instancesSelected =
        instances
        .groupBy(Constants.COL_SIGNATURE)
        .agg(aggLSH(instances.col(Constants.COL_LABEL), instances.col(Constants.COL_ID))
        .as(Constants.PICK_INSTANCE))
        .drop(Constants.COL_SIGNATURE)

      // se realiza un explode pues la UDAF devuelve una lista por cubeta
      val explodeDF =
        instancesSelected
       .select(explode(instancesSelected(Constants.PICK_INSTANCE))
       .as(Constants.COL_ID)).distinct()

      // se realiza la seleccion de instancias realizando un join entre las muestras originales y las seleccionadas por la  UDAF
      instances
      .join(explodeDF, Constants.COL_ID)
      .dropDuplicates(Constants.COL_ID)
      .drop(Constants.COL_SIGNATURE)
  }
}

// UDAF encargada de seleccionar las intancias cuando las clases son desbalanceadas
class Agg_LSH_Is_F_Unbalanced() extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(Array(
   StructField("COL_LABEL", IntegerType),
   StructField("id", IntegerType)
  ))

  override def bufferSchema: StructType =
    StructType(Array(
      StructField("pick_legal", IntegerType),
      StructField("pick_fraude", ArrayType(IntegerType)),
      StructField("cant_legal", IntegerType)
    ))

  override def dataType: DataType = ArrayType(IntegerType)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
   buffer(0) = -1
   buffer(1) = List[Int]()
   buffer(2) = 0
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
     if ((input.getInt(0) == -1)) {
       buffer(0) = input.getInt(1)
       buffer(2) = buffer.getInt(2) + 1
     }

     if ((input.getInt(0) == 1)) {
       buffer(1) = buffer(1).asInstanceOf[Seq[Int]] :+ input.getInt(1)
     }
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getInt(0)
    if(buffer1.getInt(0) == -1){
      buffer1(0) = buffer2.getInt(0)
    }
    buffer1(1) = buffer1(1).asInstanceOf[Seq[Int]] ++ buffer2(1).asInstanceOf[Seq[Int]]
    buffer1(2) = buffer1.getInt(2) + buffer2.getInt(2)
  }

  override def evaluate(buffer: Row): Any = {
    var instance_legal = buffer(0)
    if(buffer.getInt(2) == 1) {
      instance_legal = -1
    }
    (buffer(1).asInstanceOf[Seq[Int]] :+ instance_legal).filter(_ != -1)
  }
}


class Agg_LSH_Is_F_Balanced() extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(Array(
   StructField("COL_LABEL", IntegerType),
   StructField("id", IntegerType)
  ))

  override def bufferSchema: StructType =
    StructType(Array(
      StructField("labels", MapType(IntegerType, ArrayType(IntegerType)))
    ))

  override def dataType: DataType = ArrayType(IntegerType)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
   buffer(0) = Map[Int, Array[Int]]()
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val existLabel = buffer(0).asInstanceOf[Map[Int, Seq[Int]]].keys.exists(_ == input.getInt(0))
    var newVal: Int = 1
    if (existLabel) {
      newVal = buffer(0).asInstanceOf[Map[Int, Seq[Int]]](input.getInt(0))(1) + 1
    }
    buffer(0) = buffer(0).asInstanceOf[Map[Int, Seq[Int]]] +
                (input.getInt(0) -> Array(input.getInt(1), newVal))
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1(0).asInstanceOf[Map[Int, Seq[Int]]] ++
                 buffer2(0).asInstanceOf[Map[Int, Seq[Int]]]
  }

  override def evaluate(buffer: Row): Any = {
    buffer(0).asInstanceOf[Map[Int, Seq[Int]]]
    .filter(x => x._2(1) != 1)
    .map{case (a,b) => b(0)}.toArray
  }
}
