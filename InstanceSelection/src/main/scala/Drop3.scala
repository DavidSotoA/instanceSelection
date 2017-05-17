package com.lsh

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction, Window}
import org.apache.spark.sql.functions.{explode, udf}
import org.apache.spark.sql.types._

/** @author David Augusto Soto A. augusto.soto@udea.edu.co
 *  @version 1.0
 *
 *  Esta clase realiza instance selection mediante el método de DROP3
*/

object Drop3 {

  def instanceSelection(
    instances: DataFrame,
    unbalanced: Boolean,
    k_Neighbors: Int,
    distancesIntervale: Int): DataFrame = {
    // require((k_Neighbors%2 ==1) && (k_Neighbors > 0),
            // "El numero de vecinos debe ser impar y positivo")
    require((k_Neighbors + 1 < distancesIntervale),
            "El numero de vecinos debe ser menor o igual al intervalo de distancias")

    val aggKnn = new AggKnn()

    val instancesWithInfo = instances.groupBy("signature").agg(aggKnn(instances.col("features"),
                                      instances.col("idn"), instances.col("label")).as("info"))

    val transformUDF = udf(drop3(_ : Seq[Row], distancesIntervale, unbalanced, k_Neighbors))

    val remove = instancesWithInfo.withColumn("InstancesToEliminate",
    transformUDF(instancesWithInfo.col("info"))).drop("info", "signature")

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

  def drop3(instances: Seq[Row], distancesIntervale: Int, unbalanced: Boolean, k_Neighbors: Int): Seq[Int] = {
    val label = instances.head.getInt(2)
    if (isOneClass(instances, label)) {
      return returnIfOneClass(instances, unbalanced, label)
    }
      var table = completeTable(instances, distancesIntervale,  k_Neighbors, createDataTable(instances))
      var instancesForRemove = Seq[Int]()
      var numOfInstances = table.size
      var i = 0
    while(i < numOfInstances) {
      val instance = table.getRow(i)

      val instanceId = instance.id
      val instanceAssociates = instance.associates
      var requireRemove = false

      if ((instanceId.label == -1 && unbalanced) || !unbalanced) {
        requireRemove = removeInstance(instanceId, instanceAssociates, table, instancesForRemove)
      }
      if (requireRemove) {
        table.removeRow(i)
        numOfInstances = numOfInstances - 1
        table = updateTableForRemove(instanceId.id, instanceAssociates,distancesIntervale, instances,table, instancesForRemove)
        instancesForRemove = instancesForRemove :+ instanceId.id
      } else {
        i = i + 1
      }
    }
    instancesForRemove
  }

  def updateTableForRemove(
    instanceToRemove: Int,
    instanceAssociates: Seq[Int],
    delta: Int,
    instances: Seq[Row],
    table: Table,
    instanceRemove: Seq[Int]): Table = {
    var myTable = table
    for (associate <- instanceAssociates) {
      if (!instanceRemove.contains(associate)) {
        val (index, rowOfAssociate) = table.getIndexAndRowById(associate)
        val neighborsOfAssociate = rowOfAssociate.neighbors
        var distances = rowOfAssociate.distances
        var distancesOfAssociate = distances.info
        var isEmptyDistances = distances.isUpdate
        val associatesOfAssociate = rowOfAssociate.associates
        var updateDistances = distancesOfAssociate
        var newNeighbor = null.asInstanceOf[Info]

        if(!isEmptyDistances || !updateDistances.isEmpty) {

          //preguntar si se debe recalcular distancias y hacerlo en caso de que si
          if (updateDistances.isEmpty) {
            distances = recalculateDistances(associate, delta, instances, myTable)
            updateDistances = distances.info
            isEmptyDistances = distances.isUpdate
          }

          if (!updateDistances.isEmpty) {
            newNeighbor = updateDistances.head
            updateDistances = updateDistances.drop(1)
          }

          while(instanceRemove.contains(newNeighbor.id) && (!isEmptyDistances || !updateDistances.isEmpty)) {
            if (!updateDistances.isEmpty) {
              newNeighbor = updateDistances.head
              updateDistances = updateDistances.drop(1)
            }
            //preguntar si se debe recalcular distancias y hacerlo en caso de que si
            if (updateDistances.isEmpty && !isEmptyDistances){
              distances = recalculateDistances(associate, delta, instances, myTable)
              updateDistances = distances.info
              isEmptyDistances = distances.isUpdate
            }
          }

        }

        var updateNeighbors = neighborsOfAssociate.filter(x => x.id != instanceToRemove)

        if(newNeighbor != null) {
          if(!instanceRemove.contains(newNeighbor.id)) {
            updateNeighbors = updateNeighbors :+ newNeighbor
          }
        }

        val newAssociates = associatesOfAssociate.filter(x => x != instanceToRemove)
        val dist = Distances(distances.isUpdate, distances.updateIndex, updateDistances)
        val row = new RowTable(rowOfAssociate.id,
          dist,
          updateNeighbors,
          rowOfAssociate.enemy,
          newAssociates)

        myTable.replaceRow(index, row)

        if(newNeighbor != null) {
          if(!instanceRemove.contains(newNeighbor.id)) {
            myTable = updateAssociates(rowOfAssociate.id.id, newNeighbor.id, myTable)
          }
        }

      }
    }
    myTable
  }

  def recalculateDistances(
    instanceToUpdate: Int,
    delta: Int,
    instances: Seq[Row],
    table: Table): Distances = {
      var myTable = table
      var myDelta = delta
      var noMore = false
      val (index, rowToUpdate) = myTable.getIndexAndRowById(instanceToUpdate)
      val distanceIndex = rowToUpdate.distances.updateIndex
      val totalInstance = instances.size
      val availableInstance = totalInstance - distanceIndex

      if (availableInstance <= delta) {
        myDelta = availableInstance
        noMore = true
      }

      val instance = instances.filter(x =>
        x(1).asInstanceOf[Int] == rowToUpdate.id.id)(0)(0).asInstanceOf[Vector]

      val distances = calculateDistances(myDelta, distanceIndex, instance, instances)
      val newDistanceIndex = distanceIndex + myDelta
      val newDistance = Distances(noMore, newDistanceIndex, distances)
      val newRow = RowTable(
        rowToUpdate.id,
        newDistance,
        rowToUpdate.neighbors,
        rowToUpdate.enemy,
        rowToUpdate.associates)

      myTable.replaceRow(index, newRow)
      return  newDistance
    }

  /** método para calcular las distancias de una instancia contra todas las demas
   *  @param delta: indica cuantas distancias se tomaran
   *  @param distanceIndex: Indica cual fue la ultima distancia tomada
   *  @param sample: Instancia a la cual se le calcularan las distancias
   *  @param instances: Conjunto de instancias con las cuales se calculara la distancia
   *  @return Lista con todas las distancias desde distanceIndex hasta distanceIndex + delta
  */
  def calculateDistances(
    delta: Int,
    distanceIndex: Int,
    sample: Vector,
    instances: Seq[Row]): Seq[Info] = {
      val instanceSize = instances.size
      var distances = Seq[Info]()

      for(i <- 0 to (instanceSize-1)) {
        val distance = Mathematics.distance(sample, instances(i)(0).asInstanceOf[Vector])
        if(distance != 0) {
          distances = distances :+ new Info(distance, instances(i)(1).asInstanceOf[Int],
                                          instances(i)(2).asInstanceOf[Int])
        }
      }
      if(!distances.isEmpty) {
        distances = scala.util.Sorting.stableSort(distances, (i1: Info,
                                              i2: Info) => i1.distance < i2.distance)
      }
    return distances.slice(distanceIndex, distanceIndex + delta)
  }

  def calculateAllDistances(
    instances: Seq[Row],
    sample: Vector): Seq[Info] = {
      val instanceSize = instances.size
      var distances = Seq[Info]()

      for(i <- 0 to (instanceSize-1)) {
        val distance = Mathematics.distance(sample, instances(i)(0).asInstanceOf[Vector])
        if(distance != 0) {
          distances = distances :+ new Info(distance, instances(i)(1).asInstanceOf[Int],
                                          instances(i)(2).asInstanceOf[Int])
        }
      }

      if(!distances.isEmpty) {
        distances = scala.util.Sorting.stableSort(distances, (i1: Info,
                                              i2: Info) => i1.distance < i2.distance)
      }
    return distances
  }

  def removeInstance(instanceId: Id,
    associates: Seq[Int],
    table: Table,
    instanceRemove: Seq[Int]): Boolean = {
    val knn_labels = knn(Seq(1, -1), _: Seq[Info])
    var withInstanceId: Int = 0
    var withoutInstanceId: Int = 0
    for (associate <- associates) {
      if (!instanceRemove.contains(associate)) {
        val (index, rowOfAssociate) = table.getIndexAndRowById(associate)
        val labelOfAssociate = rowOfAssociate.id.label
        val neighborsOfAssociate = rowOfAssociate.neighbors
        val neighborsOfAssociateWithoutInstance = neighborsOfAssociate.filter(x => instanceId.id != x.id)
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

  def knn(labels: Seq[Int], neighbors: Seq[Info]): Int = {
    var numOflabels = Seq.fill[Int](labels.size)(0)
    for(neighbor <- neighbors) {
      var i = 0
      for(label <- labels) {
        if (label == neighbor.label) {
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

  def findNeighbors(instances: Seq[Info],
                    k_Neighbors: Int,
                    needOrder: Boolean): Seq[Info] = {
    if (needOrder) {
      val instancesInOrder = scala.util.Sorting.stableSort(instances, (i1: Info,
                                          i2: Info) => i1.distance < i2.distance)
      return instancesInOrder.take(k_Neighbors + 1)
    }
    return instances.take(k_Neighbors + 1)
  }

  def findMyNemesis(instances: Seq[Info],
                    myLabel: Int,
                    needOrder: Boolean): Double = {
    val myEnemies = killFriends(instances, myLabel)

    if(needOrder) {
      var enemiesInOrder = scala.util.Sorting.stableSort(myEnemies, (i1: Info,
                                          i2: Info) => i1.distance < i2.distance)
      return enemiesInOrder.head.distance
    }
    return myEnemies.head.distance
  }

  def killFriends(instances: Seq[Info], myLabel: Int): Seq[Info] = {
    instances.filter(x => (x.label != myLabel))
  }

  def completeTable(instances: Seq[Row], distancesIntervale: Int, k_Neighbors: Int, table: Table): Table = {
    var myTable = table
    var currentInstance: (Vector, Int, Int) = null.asInstanceOf[(Vector, Int, Int)]
    val instanceSize = instances.size
    for(i <- 0 to (instanceSize-1)) {
      currentInstance = (instances(i)(0).asInstanceOf[Vector],
                         instances(i)(1).asInstanceOf[Int],
                         instances(i)(2).asInstanceOf[Int])
     val instancesId = Id(currentInstance._2, currentInstance._3)
     var distancesOfCurrentInstance = calculateAllDistances(instances, currentInstance._1)
     val myEnemy = findMyNemesis(distancesOfCurrentInstance, currentInstance._3, false)
     distancesOfCurrentInstance = distancesOfCurrentInstance.slice(0, distancesIntervale)
     val myNeighbors = findNeighbors(distancesOfCurrentInstance, k_Neighbors, false)

     val (index, row) = myTable.getIndexAndRowById(instancesId.id)
     var isUpdateDistance = false
     if (instances.size <= distancesIntervale) {
       isUpdateDistance = true
     }
     val newRow = RowTable(instancesId,
                           Distances(isUpdateDistance, distancesIntervale, distancesOfCurrentInstance.drop(k_Neighbors + 1)),
                           myNeighbors,
                           myEnemy,
                           row.associates)
     myTable.replaceRow(index, newRow)

     for(neighbor <- myNeighbors) {
       myTable = updateAssociates(instancesId.id, neighbor.id, myTable)
     }
    }
    myTable.orderByEnemy
    myTable
  }

  def createDataTable(instances: Seq[Row]): Table = {
    val dist_null = null.asInstanceOf[Distances]
    val neighbors_null =  null.asInstanceOf[Seq[Info]]
    val enemy_null = null.asInstanceOf[Double]
    val associates_null = Seq.empty[Int]

    val instanceSize = instances.size
    val table = new Table()
    var row = null.asInstanceOf[RowTable]
    for(i <- 0 to (instanceSize-1)) {
      val id_tuple = Id(instances(i)(1).asInstanceOf[Int], instances(i)(2).asInstanceOf[Int])
      row = new RowTable(id_tuple, dist_null, neighbors_null, enemy_null, associates_null)
      table.addRow(row)
    }
    table
  }

def updateAssociates(associate: Int, instanceForUpdate: Int, table: Table): Table = {
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
