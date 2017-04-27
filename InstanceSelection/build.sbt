name := "Instance selection"

version := "1.0"

scalaVersion := "2.11.8"

test in assembly := {}

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"  % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0" % "provided"
libraryDependencies += "com.github.martincooper" %% "scala-datatable" % "0.7.0"
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.0" % Test
