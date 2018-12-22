import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Binarizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Note: The following code has been adapted from the Apache Spark ML Reference:
//https://spark.apache.org/docs/latest/ml-clustering.html#k-means
object KMeansExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("K-Means")
      .getOrCreate()


    //val year = 2009

    //Stats will be written to text file on local machine
    //val pw = new PrintWriter(new File("D:/K-MeansSummary.txt"))

    //pw.write("K-Means Statistics\r\n")



      //For usage on the googlecloud cluster.
      var traindata = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
        .load("D:/us_weather_clean/us_weather_clean.csv")

      //For usage on local windows machine
      /*var data = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
        .load("D:/data/us_weather_data_clean/us_weather_" + year + "_clean/us_weather_" + year + "_clean.csv")*/

      val indexer = new StringIndexer().setInputCol("R").setOutputCol("label")

      val assembler = new VectorAssembler()
      .setInputCols(Array("YEARMODA", "DEW", "TEMP", "PRCPN", "SLVP", "STAP", "VISB"))
      .setOutputCol("features")


      val df1 = assembler.transform(traindata)

      df1.show(false)

      val df2 = indexer.fit(df1).transform(df1)
      df2.show

      //val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), seed = 1234L)

      val kmeans = new KMeans().setK(6).setSeed(4L)
      val model = kmeans.fit(df2)

      var testdata = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
      .load("D:/us_weather_2018_clean/us_weather_2018_clean.csv")

      val indexer2 = new StringIndexer().setInputCol("R").setOutputCol("label")

    //The following columns will be considered, YEARMODA, TEMP, DEWP, PRCP, SLP, STP, VISIB
      val assembler2 = new VectorAssembler()
      .setInputCols(Array("YEARMODA", "DEW", "TEMP", "PRCPN", "SLVP", "STAP", "VISB"))
      .setOutputCol("features")

      val df3 = assembler2.transform(testdata)

      val df4 = indexer2.fit(df3).transform(df3)

      val predicitveModel = model.transform(df4)

      //val predictions = model.transform(testdata)
     // predicitveModel.describe("label", "prediction").show(false)

      val evaluator = new ClusteringEvaluator()
      .setPredictionCol("prediction")
      .setFeaturesCol("features")
      .setMetricName("silhouette")

      val silhouette = evaluator.evaluate(predicitveModel)
      println(s"Silhouette with squared euclidean distance = $silhouette")

      // Shows the result.
      println("Cluster Centers: ")
      model.clusterCenters.foreach(println)




     /* pw.write("Year: " + year + "\r\n")
      pw.write(s"Silhouette with squared euclidean distance = $silhouette\r\n")*/


    //pw.close()

  }

}
