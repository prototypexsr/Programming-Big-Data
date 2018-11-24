import org.apache.spark.sql.{SQLContext, SparkSession}
object HeatWave {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("HottestWeatherUSA")
      .getOrCreate()

    val year = 2014


    for (year <- 2014 to 2018) {
      val df = spark.read.option("header", "true").csv("D:/data/us_weather_" + year + "_clean/us_weather_" + year +
        "_clean.csv")
      df.show()

      df.createOrReplaceTempView("CC")
      val sqlDF = spark.sql("select STNCDE, YEARMODA, TEMP, MIN, MAX, PRCP, FRSHTT" +
        " from CC where TEMP > 100.0 order by YEARMODA")
      sqlDF.show()

      sqlDF.coalesce(1).write.format("com.databricks.spark.csv").option(
        "header", "true").save("D:/data/us_weather_hot_" + year)
    }
  }

}
