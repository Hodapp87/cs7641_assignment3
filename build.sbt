name := "Chris Hodapp, CS7641, assignment 3 (Unsupervised Learning)"

version := "1.0"

scalaVersion := "2.10.5"

resolvers += "Unidata maven repository" at
"http://artifacts.unidata.ucar.edu/content/repositories/unidata-releases"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1",
  "org.apache.spark" %% "spark-mllib" % "1.6.1",
  "edu.ucar" % "netcdf4" % "4.5.5" exclude("commons-logging", "commons-logging"),
  "edu.ucar" % "cdm" % "4.5.5" exclude("commons-logging", "commons-logging")
  //"edu.ucar" % "grib" % "4.5.5" exclude("commons-logging", "commons-logging")  
  //"com.github.tototoshi" %% "scala-csv" % "1.3.0"
)

// I don't understand why this is needed, but I get some
// incomprehensible JSON error otherwise despite not using JSON
// dependencyOverrides ++= Set( 
//   "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
// )
