package com.lsh

import com.github.martincooper.datatable.{DataColumn, DataRow, DataTable, DataValue}
import com.github.martincooper.datatable.DataSort.SortEnum.Descending

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction, Window}
import org.apache.spark.sql.functions.{explode, udf}
import org.apache.spark.sql.types._

case class Drop3() extends LogHelper {

  def instanceSelection(
    instances: DataFrame,
    unbalanced: Boolean,
    k_Neighbors: Int,
    distancesIntervale: Int): DataFrame = {
    // require((k_Neighbors%2 ==1) && (k_Neighbors > 0),
            // "El numero de vecinos debe ser impar y positivo")
    val aggKnn = new AggKnn()

    logger.info("..........Iniciando UDAF...............")
    val instancesWithInfo = instances.groupBy("signature").agg(aggKnn(instances.col("features"),
                                      instances.col("idn"), instances.col("label")).as("info"))
    logger.info("..........UDAF terminada...............")

    val transformUDF = udf(drop3(_ : Seq[Row], unbalanced, k_Neighbors))

    logger.info("..........Iniciando DROP3...............")
    val remove = instancesWithInfo.withColumn("InstancesToEliminate",
    transformUDF(instancesWithInfo.col("info"))).drop("info", "signature")
    logger.info("..........DROP3 terminado...............")

    logger.info("..........Eliminando instancias...............")
    val explodeDF = remove.select(explode(remove("InstancesToEliminate")).as("idn"))
    instances.join(explodeDF, Seq("idn"), "leftanti")
  }

  def eliminateInstances(instances: Seq[Row], instancesForRemove: Seq[Int]): Seq[Row] = {
    instances.filter(x => !instancesForRemove.contains(x.getInt(1)))
  }

  def isOneClass(instances: Seq[Row], label: Int): Boolean = {
    val label = instances.head.getInt(2)
    !instances.exists(x => x.getInt(2) != label)
  }

  def returnIfOneClass(instances: Seq[Row], unbalanced: Boolean, label: Int): Seq[Int] = {
    if(!unbalanced || (unbalanced && label == -1)){
      return instances.drop(1).map(_.getInt(1))
    }else {
      return List(): Seq[Int]
    }
  }

  def drop3(instances: Seq[Row], unbalanced: Boolean, k_Neighbors: Int): Seq[Int] = {
    val label = instances.head.getInt(2)
    if (isOneClass(instances, label)) {
      return returnIfOneClass(instances, unbalanced, label)
    }
    var table = completeTable(instances, k_Neighbors, createDataTable(instances))
    var instancesForRemove = Seq[Int]()
    var numOfInstances = table.size
    var i = 0
    while(i < numOfInstances) {
      val instance = table(i)
      val instanceRow = new RowTable(instance(0).asInstanceOf[(Int, Int)],
                              instance(1).asInstanceOf[Seq[(Double, Int, Int)]],
                              instance(2).asInstanceOf[Seq[(Double, Int, Int)]],
                              instance(3).asInstanceOf[Double],
                              instance(4).asInstanceOf[Seq[Int]])
      val instanceId = instanceRow.id
      val instanceAssociates = instanceRow.associates
      var requireRemove = false

      if ((instanceId._2 == -1 && unbalanced) || !unbalanced) {
        requireRemove = removeInstance(instanceId, instanceAssociates, table, instancesForRemove)
      }
      if (requireRemove) {
        table = table.rows.remove(i).get
        numOfInstances = numOfInstances - 1
        table = updateTableForRemove(instanceId._1, instanceAssociates, table, instancesForRemove)
        instancesForRemove = instancesForRemove :+ instanceId._1
      } else {
        i = i + 1
      }
    }
    instancesForRemove
  }

  def updateTableForRemove(instanceToRemove: Int,
    instanceAssociates: Seq[Int],
    table: DataTable,
    instanceRemove: Seq[Int]): DataTable = {
    var mytable = table
    for (associate <- instanceAssociates) {
      if (!instanceRemove.contains(associate)) {
        val (index, rowOfAssociate) = getIndexAndRowById(associate, table)
        val neighborsOfAssociate = rowOfAssociate.neighbors
        val distancesOfAssociate = rowOfAssociate.distances
        val associatesOfAssociate = rowOfAssociate.associates
        var updateDistances = distancesOfAssociate
        var newNeighbor = null.asInstanceOf[(Double, Int, Int)]

        if(!updateDistances.isEmpty) {
          newNeighbor = distancesOfAssociate.head
          updateDistances = distancesOfAssociate.drop(1)
          while(instanceRemove.contains(newNeighbor._2) && !updateDistances.isEmpty)  {
            newNeighbor = updateDistances.head
            updateDistances = updateDistances.drop(1)
          }
        }

        var updateNeighbors = neighborsOfAssociate.filter(x => x._2 != instanceToRemove)
        if(!updateDistances.isEmpty) {
         updateNeighbors = updateNeighbors :+ newNeighbor
        }

        val newAssociates = associatesOfAssociate.filter(x => x != instanceToRemove)
        mytable = mytable.rows.replace(index, DataValue.apply(rowOfAssociate.id),
                                    DataValue.apply(updateDistances),
                                    DataValue.apply(updateNeighbors),
                                    DataValue.apply(rowOfAssociate.enemy),
                                    DataValue.apply(newAssociates)).get
        if(!updateDistances.isEmpty) {
          mytable = updateAssociates(rowOfAssociate.id._1, newNeighbor._2, mytable)
        }
      }
    }
    mytable
  }

  def removeInstance(instanceId: (Int, Int),
    associates: Seq[Int],
    table: DataTable,
    instanceRemove: Seq[Int]): Boolean = {
    val knn_labels = knn(Seq(1, -1), _: Seq[(Double, Int, Int)])
    var withInstanceId: Int = 0
    var withoutInstanceId: Int = 0
    for (associate <- associates) {
      if (!instanceRemove.contains(associate)) {
        val (index, rowOfAssociate) = getIndexAndRowById(associate, table)
        val labelOfAssociate = rowOfAssociate.id._2
        val neighborsOfAssociate = rowOfAssociate.neighbors
        val neighborsOfAssociateWithoutInstance = neighborsOfAssociate.filter(x => instanceId._1 != x._2)
        val knnWithInstance = knn_labels(neighborsOfAssociate)
        val knnWithoutInstance = knn_labels(neighborsOfAssociateWithoutInstance)
        if (knnWithInstance == labelOfAssociate) {
          withInstanceId = withInstanceId + 1
        }
        if (knnWithoutInstance == labelOfAssociate) {
          withoutInstanceId = withoutInstanceId + 1
        }
      }
    }
    if (withoutInstanceId >= withInstanceId) {
      return true
    }
    return false
  }

  def knn(labels: Seq[Int], neighbors: Seq[(Double, Int, Int)]): Int = {
    var numOflabels = Seq.fill[Int](labels.size)(0)
    for(neighbor <- neighbors) {
      var i = 0
      for(label <- labels) {
        if (label == neighbor._3) {
          numOflabels = numOflabels.updated(i, numOflabels(i) + 1)
        }
        i = i + 1
      }
    }
    if(!numOflabels.exists(x => x != numOflabels(0))) {
        return 0
    }
    return labels.zip(numOflabels).maxBy(x => x._2)._1
  }

  def createDataTable(instances: Seq[Row]): DataTable = {
    val id_col = new DataColumn[(Int, Int)]("id_col", Array[(Int, Int)]())
    val dist_col = new DataColumn[Seq[(Double, Int, Int)]]("Distancias",
                                  Array[Seq[(Double, Int, Int)]]())
    val neighbors_col = new DataColumn[Seq[(Double, Int, Int)]]("Vecinos",
                                       Array[Seq[(Double, Int, Int)]]())
    val enemy_col = new DataColumn[Double]("Enemigo", Array[Double]())
    val associates_col = new DataColumn[Seq[Int]]("Asociados", Array[Seq[Int]]())
    var table = DataTable("NewTable", Seq(id_col, dist_col, neighbors_col, enemy_col,
                                          associates_col))

    val dist_null = DataValue.apply(null.asInstanceOf[Seq[(Double, Int, Int)]])
    val neighbors_null = dist_null
    val enemy_null = DataValue.apply(null.asInstanceOf[Double])
    val associates_null = DataValue.apply(Seq.empty[Int])

    val instanceSize = instances.size
    for(i <- 0 to (instanceSize-1)) {
      val id_tuple = (instances(i)(1).asInstanceOf[Int], instances(i)(2).asInstanceOf[Int])
      val value_id_tuple = DataValue.apply(id_tuple)
      val value_null = DataValue.apply(null)
      table = table.get.rows.add(value_id_tuple, dist_null, neighbors_null, enemy_null,
                                 associates_null)
    }
    table.get
  }

  def completeTable(instances: Seq[Row], k_Neighbors: Int, table: DataTable): DataTable = {
    var myTable = table
    var currentInstance: (Vector, Int, Int) = null.asInstanceOf[(Vector, Int, Int)]
    val instanceSize = instances.size
    for(i <- 0 to (instanceSize-1)) {
      currentInstance = (instances(i)(0).asInstanceOf[Vector],
                         instances(i)(1).asInstanceOf[Int],
                         instances(i)(2).asInstanceOf[Int])
      val instancesId = (currentInstance._2, currentInstance._3)
      val distancesOfCurrentInstance = calculateDistances(currentInstance._1, instances)
      val myNeighbors = findNeighbors(distancesOfCurrentInstance, k_Neighbors, false)
      val myEnemy = findMyNemesis(distancesOfCurrentInstance, currentInstance._3, false)

      // Traer indice y row de la instancia
      val (index, row) = getIndexAndRowById(instancesId._1, myTable)
      myTable = myTable.rows.replace(index, DataValue.apply(instancesId),
                                            DataValue.apply(
                                              distancesOfCurrentInstance.drop(k_Neighbors + 1)),
                                            DataValue.apply(myNeighbors),
                                            DataValue.apply(myEnemy),
                                            DataValue.apply(row.associates)).get
      for(neighbor <- myNeighbors) {
        myTable = updateAssociates(instancesId._1, neighbor._2, myTable)
      }
    }
    myTable.quickSort("Enemigo", Descending).get.toDataTable
  }

  def updateAssociates(associate: Int, instanceForUpdate: Int, table: DataTable): DataTable = {
    var myTable = table
    val (index, row) = getIndexAndRowById(instanceForUpdate, myTable)
    myTable.rows.replace(index, DataValue.apply(row.id),
                                DataValue.apply(row.distances),
                                DataValue.apply(row.neighbors),
                                DataValue.apply(row.enemy),
                                DataValue.apply(row.associates :+ associate)).get
  }


  def getIndexAndRowById(id: Int, table: DataTable): (Int, RowTable) = {
   val index = table.indexWhere(x => x.values(0).asInstanceOf[(Int, Int)]._1 == id)
   val row = table.filter(x => x.values(0).asInstanceOf[(Int, Int)]._1 == id).toDataTable(0)
   val rowT = new RowTable(row(0).asInstanceOf[(Int, Int)],
                           row(1).asInstanceOf[Seq[(Double, Int, Int)]],
                           row(2).asInstanceOf[Seq[(Double, Int, Int)]],
                           row(3).asInstanceOf[Double],
                           row(4).asInstanceOf[Seq[Int]])
   (index, rowT)
 }

  def findNeighbors(instances: Seq[(Double, Int, Int)],
                    k_Neighbors: Int,
                    needOrder: Boolean): Seq[(Double, Int, Int)] = {
    if (needOrder) {
      val instancesInOrder = scala.util.Sorting.stableSort(instances, (i1: (Double, Int, Int),
                                          i2: (Double, Int, Int)) => i1._1 < i2._1)
      return instancesInOrder.take(k_Neighbors + 1)
    }
    return instances.take(k_Neighbors + 1)
  }

  def findMyNemesis(instances: Seq[(Double, Int, Int)],
                    myLabel: Int,
                    needOrder: Boolean): Double = {
    val myEnemies = killFriends(instances, myLabel)

    if(needOrder) {
      var enemiesInOrder = scala.util.Sorting.stableSort(myEnemies, (i1: (Double, Int, Int),
                                          i2: (Double, Int, Int)) => i1._1 < i2._1)
      return enemiesInOrder.head._1
    }
    return myEnemies.head._1
  }

  def killFriends(instances: Seq[(Double, Int, Int)], myLabel: Int): Seq[(Double, Int, Int)] = {
    instances.filter(x => (x._3 != myLabel))
  }

  def calculateDistances(sample: Vector, instances: Seq[Row]): Seq[(Double, Int, Int)] = {
    val instanceSize = instances.size
    var distances = Array[(Double, Int, Int)]()
    for(i <- 0 to (instanceSize-1)) {
      val distance = Mathematics.distance(sample, instances(i)(0).asInstanceOf[Vector])
      if(distance != 0) {
        distances = distances :+ (distance, instances(i)(1).asInstanceOf[Int],
                                  instances(i)(2).asInstanceOf[Int])
      }
    }
    if(!distances.isEmpty) {
      distances = scala.util.Sorting.stableSort(distances, (i1: (Double, Int, Int),
                                            i2: (Double, Int, Int)) => i1._1 < i2._1)
    }
    distances
  }

  def calculateDistances2(
    sample: Vector,
    instances: Seq[Row],
    distancesIntervale: Int,
    initDistance : Int): Seq[(Double, Int, Int)] = {
      val instanceSize = instances.size
      var distances = Array[(Double, Int, Int)]()
      for(i <- 0 until instanceSize) {
        val distance = Mathematics.distance(sample, instances(i)(0).asInstanceOf[Vector])
        if(distance != 0) {
          distances = distances :+ (distance, instances(i)(1).asInstanceOf[Int],
                                    instances(i)(2).asInstanceOf[Int])
        }
      }
      if(!distances.isEmpty) {
        distances = scala.util.Sorting.stableSort(distances, (i1: (Double, Int, Int),
                                              i2: (Double, Int, Int)) => i1._1 < i2._1)
      }
      distances
  }

  def completeTable2(instances: Seq[Row], k_Neighbors: Int, table: Table): Table = {
    var myTable = table
    var currentInstance: (Vector, Int, Int) = null.asInstanceOf[(Vector, Int, Int)]
    val instanceSize = instances.size
    for(i <- 0 to (instanceSize-1)) {
      println(i)
      currentInstance = (instances(i)(0).asInstanceOf[Vector],
                         instances(i)(1).asInstanceOf[Int],
                         instances(i)(2).asInstanceOf[Int])
     val instancesId = (currentInstance._2, currentInstance._3)
     val distancesOfCurrentInstance = calculateDistances(currentInstance._1, instances)
     val myNeighbors = findNeighbors(distancesOfCurrentInstance, k_Neighbors, false)
     val myEnemy = findMyNemesis(distancesOfCurrentInstance, currentInstance._3, false)

     val (index, row) = myTable.getIndexAndRowById(instancesId._1)
     val newRow = RowTable(instancesId,
                           distancesOfCurrentInstance.drop(k_Neighbors + 1),
                           myNeighbors,
                           myEnemy,
                           row.associates)
     myTable.replaceRow(index, newRow)

     for(neighbor <- myNeighbors) {
       myTable = updateAssociates2(instancesId._1, neighbor._2, myTable)
     }
    }
    myTable.orderByEnemy
    myTable
  }

  def createDataTable2(instances: Seq[Row]): Table = {
    val dist_null = null.asInstanceOf[Seq[(Double, Int, Int)]]
    val neighbors_null = dist_null
    val enemy_null = null.asInstanceOf[Double]
    val associates_null = Seq.empty[Int]

    val instanceSize = instances.size
    val table = new Table()
    var row = null.asInstanceOf[RowTable]
    for(i <- 0 to (instanceSize-1)) {
      val id_tuple = (instances(i)(1).asInstanceOf[Int], instances(i)(2).asInstanceOf[Int])
      row = new RowTable(id_tuple, dist_null, neighbors_null, enemy_null, associates_null)
      table.addRow(row)
    }
    table
  }

def updateAssociates2(associate: Int, instanceForUpdate: Int, table: Table): Table = {
    var myTable = table
    val (index, row) = myTable.getIndexAndRowById(instanceForUpdate)
    val newRow = RowTable(row.id,
                          row.distances,
                          row.neighbors,
                          row.enemy,
                          row.associates :+ associate)
    myTable.replaceRow(index, newRow)
    myTable
  }

}

class AggKnn() extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = StructType(Array(
    StructField("features", VectorType),
    StructField("idn", IntegerType),
    StructField("label", IntegerType)
  ))

   override def bufferSchema: StructType = StructType(Array(
     StructField("allInfo", ArrayType(StructType(Array(
                                       StructField("features", VectorType),
                                       StructField("idn", IntegerType),
                                       StructField("label", IntegerType)
                                     ))))
   ))

   override def dataType: DataType = ArrayType(StructType(Array(
                                     StructField("features", VectorType),
                                     StructField("idn", IntegerType),
                                     StructField("label", IntegerType)
                                    )))

   override def deterministic: Boolean = true

   override def initialize(buffer: MutableAggregationBuffer): Unit = {
     buffer(0) = Array[(Vector, Int, Int)]()
   }

   override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
     buffer(0) = buffer(0).asInstanceOf[Seq[(Vector, Int, Int)]] :+
        (input(0).asInstanceOf[Vector], input(1).asInstanceOf[Int], input(2).asInstanceOf[Int])
   }

   override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
     buffer1(0) = buffer1(0).asInstanceOf[Seq[(Vector, Int, Int)]] ++
                  buffer2(0).asInstanceOf[Seq[(Vector, Int, Int)]]
   }

   override def evaluate(buffer: Row): Any = {
     buffer(0)
   }
 }
