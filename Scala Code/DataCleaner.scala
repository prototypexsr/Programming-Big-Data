import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

import scala.collection.mutable.HashMap

object DataCleaner {

  val hashMap1: HashMap[String, String] = HashMap(("SNDP", "999.9"), ("MIN", "9999.9"), ("MAX", "9999.9"),
    ("WDSP", "999.9"), ("MXSPD", "999.9"), ("VISIB", "999.9"), ("DEWP", "9999.9"), ("SLP", "9999.9"),
    ("STP", "9999.9"), ("PRCP", "99.99"), ("GUST", "999.9"))

  def clean(sampleDF: DataFrame): DataFrame = {
    var df = sampleDF

    df = df.withColumn("SWNDP", regexp_replace(col("SNDP"), "999.9", df.filter(df("SNDP") =!= hashMap1
    ("SNDP")).agg(avg(col("SNDP"))).head().getDouble(0).toString()))
    df = df.withColumn("MINIMUM", regexp_replace(col("MIN"), "9999.9", df.filter(df("MIN") =!= hashMap1("MIN"))
      .agg(avg(col("MIN"))).head().getDouble(0).toString()))
    df = df.withColumn("MAXIMUM", regexp_replace(col("MAX"), "9999.9", df.filter(df("MAX") =!= hashMap1("MAX"))
      .agg(avg(col("MAX"))).head().getDouble(0).toString()))
    df = df.withColumn("WINDSPD", regexp_replace(col("WDSP"), "999.9", df.filter(df("WDSP") =!= hashMap1("WDSP"))
      .agg(avg(col("WDSP"))).head().getDouble(0).toString()))
    df = df.withColumn("VISB", regexp_replace(col("VISIB"), "999.9", df.filter(df("VISIB") =!= hashMap1("VISIB"))
      .agg(avg(col("VISIB"))).head().getDouble(0).toString()))
    df = df.withColumn("DEW", regexp_replace(col("DEWP"), "9999.9", df.filter(df("DEWP") =!= hashMap1("DEWP"))
      .agg(avg(col("DEWP"))).head().getDouble(0).toString()))
    df = df.withColumn("GUSTSPD", regexp_replace(col("GUST"), "999.9", df.filter(df("GUST") =!= hashMap1("GUST"))
      .agg(avg(col("GUST"))).head().getDouble(0).toString()))
    df = df.withColumn("SLVP", regexp_replace(col("SLP"), "9999.9", df.filter(df("SLP") =!= hashMap1("SLP")).agg
    (avg(col("SLP"))).head().getDouble(0).toString()))
    df = df.withColumn("STAP", regexp_replace(col("STP"), "9999.9", df.filter(df("STP") =!= hashMap1("STP")).agg
    (avg(col("STP"))).head().getDouble(0).toString()))
    df = df.withColumn("PRCPN", regexp_replace(col("PRCP"), "99.99", df.filter(df("PRCP") =!= hashMap1("PRCP"))
      .agg(avg(col("PRCP"))).head().getDouble(0).toString()))
    df = df.withColumn("MAXSPD", regexp_replace(col("MXSPD"), "999.99", df.filter(df("MXSPD") =!= hashMap1("MXSPD"))
      .agg(avg(col("MXSPD"))).head().getDouble(0).toString()))

    return df
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[3]")
      .appName("WeatherUSA")
      .getOrCreate()

    val year = 2010
    var df1 = spark.read.option("header", "true").option("inferSchema", "true").csv("D:/data/us_weather_2009.csv")
    df1 = clean(df1)



    var counter = 0
    for (year <- 2010 to 2017) {
      var df2 = spark.read.option("header", "true").option("inferSchema", "true").csv("D:/data/us_weather_" + year +
        ".csv")

      df2 = clean(df2)
      df1 = df1.union(df2)


    }

    df1.createOrReplaceTempView("USW")


    var sqlDF = spark.sql("select STNCDE, YEARMODA  " +
      ", TEMP, DEW, SLVP, " +
      "STAP, VISB, " +
      "WINDSPD, " +
      "MXSPD, GUSTSPD, MINIMUM, MAXIMUM," +
      "PRCPN, " +
      "F, R, S, H, Th, To" +
      " from USW")


    sqlDF.coalesce(1).write.format("com.databricks.spark.csv").option(
      "header", "true").save("D:/us_weather_clean.csv")

    var df3 = spark.read.option("header", "true").option("inferSchema", "true").csv("D:/data/us_weather_2018.csv")

    df3 = clean(df3)

    df3.createOrReplaceTempView("USW")


    var sqlDF2 = spark.sql("select STNCDE, YEARMODA  " +
      ", TEMP, DEW, SLVP, " +
      "STAP, VISB, " +
      "WINDSPD, " +
      "MXSPD, GUSTSPD, MINIMUM, MAXIMUM," +
      "PRCPN, " +
      "F, R, S, H, Th, To" +
      " from USW")


    sqlDF2.coalesce(1).write.format("com.databricks.spark.csv").option(
      "header", "true").save("D:/us_weather_clean_2018.csv")


  }


}

