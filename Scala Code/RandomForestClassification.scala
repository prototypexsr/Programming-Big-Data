import java.io.{File, PrintWriter}

import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


//Note: The following code has been adapted from the Apache Spark ML Reference:
//https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-regression

object RandomForestClassification {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("RandomForest")
      .getOrCreate()

    val sqlContext = spark.sqlContext; import sqlContext.implicits._

    val pw = new PrintWriter(new File("D:/data/RandomForestSummary.txt"))
    pw.write("Random Forest Statistics\r\n")
    pw.write("Number of Trees: 10\r\n")


     var traindata = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
      .load("D:/us_weather_clean/us_weather_clean.csv")



      var vassembler = new VectorAssembler().setInputCols(Array("YEARMODA", "DEW", "TEMP", "PRCPN", "SLVP", "STAP", "VISB"))
        .setOutputCol("PRCPV")

      traindata = vassembler.transform(traindata)


      val labelIndexer = new StringIndexer().setInputCol("R").setOutputCol("indexedLabel").fit(traindata)

      val featureIndexer = new VectorIndexer().setInputCol("PRCPV").setOutputCol("indexedFeatures").fit(traindata)

      val assembler = new VectorAssembler()
        .setInputCols(Array("indexedFeatures"))
        .setOutputCol("features")


      val rf = new RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("features")
        .setNumTrees(10)

      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)

      val pipeline = new Pipeline()
        .setStages(Array(labelIndexer, featureIndexer, assembler, rf, labelConverter))

      val model = pipeline.fit(traindata)


      var testdata = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
      .load("D:/us_weather_2018_clean/us_weather_2018_clean.csv")


    var vassembler2 = new VectorAssembler().setInputCols(Array("YEARMODA", "DEW", "TEMP", "PRCPN", "SLVP", "STAP", "VISB"))
      .setOutputCol("PRCPV")

    testdata = vassembler2.transform(testdata)

    val labelIndexer2 = new StringIndexer().setInputCol("R").setOutputCol("indexedLabel").fit(testdata)

    val featureIndexer2 = new VectorIndexer().setInputCol("PRCPV").setOutputCol("indexedFeatures").fit(testdata)

    val assembler2 = new VectorAssembler()
      .setInputCols(Array("indexedFeatures"))
      .setOutputCol("features")

    val labelConverter2 = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer2.labels)

    val pipeline2 = new Pipeline()
      .setStages(Array(labelIndexer2, featureIndexer2, assembler2, rf, labelConverter2))

      // Make predictions.
      val predictions = model.transform(testdata)

      // Select example rows to display.
      predictions.select("predictedLabel", "PRCPV", "features").show(30)

      // Select (prediction, true label) and compute test error.
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

      val accuracy = evaluator.evaluate(predictions)
      val lp = predictions.select("indexedLabel", "prediction")
      val counttotal = predictions.count()
      val correct = lp.filter($"indexedLabel" === $"prediction").count()
      val wrong = lp.filter(($"indexedLabel" =!= $"prediction")).count()
      val truep = lp.filter(($"prediction" === 1.0)  && ($"indexedLabel" === 1.0)).count()
      val truen = lp.filter(($"prediction" === 0.0)  && ($"indexedLabel" === 0.0)).count()
      val falseN = lp.filter(($"prediction" === 0.0) && ($"indexedLabel" === 1.0)).count()
      val falseP = lp.filter(($"prediction" === 1.0) && ($"indexedLabel" === 0.0)).count()
      val ratioWrong = wrong.toFloat / counttotal.toFloat
      val ratioRight = correct.toFloat / counttotal.toFloat
      val ones = lp.filter($"prediction" === 1.0).count()
      val zeroes = lp.filter($"prediction" === 0.0).count()
        println(s"Test Error = ${(1.0 - accuracy)}")

    val  predictionAndLabels = predictions.select("rawPrediction", "indexedLabel").rdd.map(x => (x(0).asInstanceOf[DenseVector](1), x(1).asInstanceOf[Double]))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    println("area under the precision-recall curve: " + metrics.areaUnderPR)
    println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)


      pw.write(s"Total # of Predictions: ${counttotal} \r\n")
      pw.write(s"# of Correct Predictions: ${correct} \r\n")
      pw.write(s"# of Wrong Predictions: ${wrong} \r\n")
      pw.write(s"# of True Negatives: ${truen} \r\n")
      pw.write(s"# of True Positives: ${truep} \r\n")
      pw.write(s"# of False Negatives: ${falseN} \r\n")
      pw.write(s"# of False Positives: ${falseP} \r\n")
      pw.write(s"Error Ratio: ${ratioWrong} \r\n")
      pw.write(s"Accuracy: ${ratioRight} \r\n")
      pw.write(s"Ones: ${ones} \r\n")
      pw.write(s"Zeroes: ${zeroes} \r\n")
      pw.write("Area under the precision-recall curve: " + metrics.areaUnderPR + "\r\n")
      pw.write("Area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC + "\r\n")



      pw.close()

     metrics.roc().coalesce(1).saveAsTextFile("D:/Scatter2")

      val rfModel = model.stages(3).asInstanceOf[RandomForestClassificationModel]
      println(s"Learned classification forest model:\n ${rfModel.toDebugString}")


      /*pw.write("Year: " + year + "\r\n")
      pw.write(s"Test Model accuracy: $accuracy\r\n")*/


    //pw.close()


  }

}
