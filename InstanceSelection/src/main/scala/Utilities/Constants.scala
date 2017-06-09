package utilities

object Constants extends java.io.Serializable {
  val APP_NAME = "Instance_selection"
  val MASTER = "local"
  val LSH_HYPERPLANES_METHOD = "hyperplanes"
  val LSH_PROJECTION_METHOD = "projection"
  val INSTANCE_SELECTION_ENTROPY_METHOD = "entropia"
  val INSTANCE_SELECTION_DROP3_METHOD = "drop3"
  val INSTANCE_SELECTION_LSH_IS_S_METHOD = "lsh_is_s"
  val INSTANCE_SELECTION_LSH_IS_F_METHOD = "lsh_is_f"
  val FORMAT_PARQUET= "parquet"
  val COL_FEATURES = "features"
  val COL_SIGNATURE = "signature"
  val COL_SCALED = "scaled"
  val COL_ENTROPY = "entropy"
  val COL_ROULETTE = "roulette"
  val COL_ID = "idn"
  val COL_LABEL = "label"
  val PICK_INSTANCE = "pick_instance"
  val PICK_CONCAT = "pick_concat"
  val SPARK_SESSION_MODE_LOCAL = "local"
  val SPARK_SESSION_MODE_CLUSTER = "cluster"
}
