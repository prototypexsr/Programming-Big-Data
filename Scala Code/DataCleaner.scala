import org.apache.spark.sql.{SQLContext, SparkSession}

object DataCleaner {
    def main(args: Array[String]): Unit = {
      val spark = SparkSession.builder
        .master("local[*]")
        .appName("WeatherUSA")
        .getOrCreate()

      val year = 2014


      for ( year <- 2014 to 2018) {
        val df = spark.read.option("header", "true").csv("D:/data/us_weather_" + year + ".csv")
        df.show()

        df.createOrReplaceTempView("CC")
        val sqlDF = spark.sql("select STNCDE, YEARMODA, TEMP, DEWP, SLP, STP, VISIB, WDSP, MXSPD, GUST, MIN, MAX," +
          "PRCP," +
          "SNDP, FRSHTT" +
          " from CC where (STNCDE NOT BETWEEN 704898 AND 711680)" +
          "AND (STNCDE NOT BETWEEN 711680 AND 713407)" +
          "AND (STNCDE NOT BETWEEN 713407 AND 720000)" +
          "AND (STNCDE NOT BETWEEN 912944 AND 990062)")
        sqlDF.show()

        sqlDF.coalesce(1).write.format("com.databricks.spark.csv").option(
          "header", "true").save("D:/data/us_weather_" + year + "_clean")
      }
    }
  }

