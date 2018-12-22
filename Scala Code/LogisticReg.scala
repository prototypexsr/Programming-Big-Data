import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import java.io._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//Note: The following code has been adapted from the Apache Spark ML Reference:
//https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
object LogisticReg{
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("Logistic Regression")
      .getOrCreate()

    val sqlContext = spark.sqlContext;
    import sqlContext.implicits._
    val pw = new PrintWriter(new File("D:/data/LogisticRegSummary.txt"))

   pw.write("Logistic Regression Statistics\r\n")

    //We will be going through 10 years of data

    // Load the data stored in csv format as a DataFrame.

    var traindata = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
      .load("D:/us_weather_clean/us_weather_clean.csv")


    //We want to predict any instance of rain. The csv file has a column named R, which indicates if rainfall was
    // reported or not; values are either 1 (yes) or 0 (no)

    //The following columns will be considered, YEARMODA, TEMP, DEWP, PRCP, SLP, STP, VISIB
    val assembler = new VectorAssembler()
      .setInputCols(Array("STNCDE", "DEW", "TEMP", "PRCPN", "SLVP", "STAP", "VISB", "Th", "WINDSPD","MXSPD", "MINIMUM", "MAXIMUM" ))
      .setOutputCol("features")

    val df1 = assembler.transform(traindata)

    df1.show(false)

    val indexer = new StringIndexer().setInputCol("R").setOutputCol("label")

    val df2 = indexer.fit(df1).transform(df1)

    df2.printSchema()

    val model = new LogisticRegression()


    var testdata = spark.read.format("csv").option("header", "true").option("inferSchema", "true")
      .load("D:/us_weather_2018_clean/us_weather_2018_clean.csv")

    val df3 = assembler.transform(testdata)


    val df4 = indexer.fit(df3).transform(df3)
    df4.show(false)

    val predicitveModel = model.fit(df2).transform(df4)


    val evaluator = new BinaryClassificationEvaluator().
      setMetricName("areaUnderROC").
      setRawPredictionCol("rawPrediction").
      setLabelCol("label")

    val accuracy = evaluator.evaluate(predicitveModel)

    //Statistics of Logistic Regression

    val lp = predicitveModel.select("label", "prediction")
    val counttotal = predicitveModel.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(($"label" =!= $"prediction")).count()
    val truep = lp.filter(($"prediction" === 1.0)  && ($"label" === 1.0)).count()
    val truen = lp.filter(($"prediction" === 0.0)  && ($"label" === 0.0)).count()
    val falseN = lp.filter(($"prediction" === 0.0) && ($"label" === 1.0)).count()
    val falseP = lp.filter(($"prediction" === 1.0) && ($"label" === 0.0)).count()
    val ratioWrong = wrong.toFloat / counttotal.toFloat
    val ratioRight = correct.toFloat / counttotal.toFloat
    val ones = lp.filter($"prediction" === 1).count()
    val zeroes = lp.filter($"prediction" === 0.0).count()

    val  predictionAndLabels = predicitveModel.select("rawPrediction", "label").rdd.map(x => (x(0).asInstanceOf[DenseVector](1), x(1).asInstanceOf[Double]))
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
    pw.write(s"Test Error Percentage: ${ratioWrong} \r\n")
    pw.write(s"Accuracy: ${ratioRight} \r\n")
    pw.write(s"# of Times Algorithm Predicted 1: ${ones} \r\n")
    pw.write(s"# of Times Algorithm Predicted 0: ${zeroes} \r\n")
    pw.write("Area under the precision-recall curve: " + metrics.areaUnderPR + "\r\n")
    pw.write("Area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC + "\r\n")


    metrics.roc().coalesce(1).saveAsTextFile("D:/Scatter.txt")



    pw.close()

  }

}