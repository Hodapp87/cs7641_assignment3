#!/bin/sh

# This starts spark-shell but sets the class path to make use of the
# Scala classes and the Java dependencies.  You must have built the
# Scala code for it to be usable.
$SPARK_HOME/bin/spark-shell \
    --driver-class-path ./target/scala-2.10/classes \
    --jars ./lib/*.jar $@
