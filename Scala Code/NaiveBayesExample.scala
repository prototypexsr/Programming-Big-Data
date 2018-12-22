import java.io.{File, PrintWriter}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


//Note: The following code has been adapted from the Apache Spark ML Reference:
//https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes
object NaiveBayesExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("NaiveBayes")
      .getOrCreate()

    val sqlContext = spark.sqlContext; import sqlContext.implicits._

    val year = 2009
   /* val pw = new PrintWriter(new File("D:/NaiveBayesSummary.txt"))

    pw.write("Naive Bayes Statistics\r\n")*/

    for ( year <- 2009 to 2018) {
      // Load the data stored in LIBSVM format as a DataFrame.
     /* var data = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
        .load("D:/data/us_weather_data_clean/us_weather_" + year + "_clean/us_weather_" + year + "_clean.csv")*/

     var data = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
       .load("gs://dataproc-2edb4c03-c77c-4c73-94c5-3d77afbb94fd-us-east1/us_weather_data_clean/us_weather_" + year + "_clean/us_weather_" + year + "_clean.csv")


      data.show(false)

      data.printSchema()

      val indexer = new StringIndexer().setInputCol("R").setOutputCol("label")

      val assembler = new VectorAssembler()
        .setInputCols(Array("YEARMODA", "PRCP", "SLP", "STP", "VISIB"))
        .setOutputCol("features")


      val df1 = assembler.transform(data)

      val df2 = indexer.fit(df1).transform(df1)
      df2.show


      // Split the data into training and test sets (30% held out for testing)
      val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), seed = 1234L)


      // Train a NaiveBayes model.
      val model = new NaiveBayes().fit(trainingData)

      val predictions = model.transform(testData)
      //predictions.show(false )
      predictions.describe("label", "prediction").show(false)

      val training_results = model.transform(trainingData)

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

      val accuracy = evaluator.evaluate(predictions)

      val evaluator2 = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

      val accuracy2 = evaluator.evaluate(training_results)

      println(s"Test set accuracy = $accuracy")
      println(s"Training set accuracy = $accuracy2")


     /* pw.write("Year: " + year + "\r\n")
      pw.write(s"Test Model accuracy: $accuracy\r\n")
      pw.write(s"Training Model accuracy: $accuracy2\r\n")*/


    }
    //pw.close()


  }

}
