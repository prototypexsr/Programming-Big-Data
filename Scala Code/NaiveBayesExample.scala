import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object NaiveBayesExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("NaiveBayesExample")
      .getOrCreate()

    val sqlContext = spark.sqlContext; import sqlContext.implicits._

    val year = 2014

    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
      .load("D:/data/us_weather_2018.csv")

    val indexer = new StringIndexer().setInputCol("VISIB").setOutputCol("label")

    val assembler =  new VectorAssembler()
      .setInputCols(Array("PRCP", "YEARMODA"))
      .setOutputCol("features")




    //assembler.setOutputCol("label")

    //val assembler2 = assembler.setInputCols(Array(assembler.getOutputCol)).setOutputCol("label")

    val df1 = assembler.transform(data)
    //output.show(false)
    //df1.withColumn("label", df1("features"))

    df1.show(false)

    val df2 = indexer.fit(df1).transform(df1)
    df2.show

    //val pipeline = new Pipeline()

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), seed = 1234L)



    //trainingData.show(false)

    // Train a NaiveBayes model.
    val model = new NaiveBayes().fit(trainingData)


    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show(false )


    //.fit(trainingData)

    //val rModel = model.fit(trainingData)
    //val predicitveModel = model.fit(testData)


    // $example off$

    //spark.stop()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    //model.save("D:/data/LogisticRegression")


  }

}
