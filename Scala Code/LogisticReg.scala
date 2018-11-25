import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.classification.ProbabilisticClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
//import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object LogisticReg{
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

    val pipeline = new Pipeline()

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), seed = 1234L)

    trainingData.show(false)

    // Train a NaiveBayes model.


    // Select example rows to display.
    val model = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      //.fit(trainingData)

    val rModel = model.fit(trainingData)
    val predicitveModel = model.fit(testData)

    // $example off$

    //spark.stop()
    // Print the coefficients and intercepts for logistic regression with multinomial family
    println(s"Multinomial coefficients: ${rModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${rModel.interceptVector}")

    println(s"Multinomial coefficients: ${predicitveModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${predicitveModel.interceptVector}")

    val trainingSummary = rModel.summary

    

    val accuracy = trainingSummary.accuracy
    val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    val truePositiveRate = trainingSummary.weightedTruePositiveRate
    val fMeasure = trainingSummary.weightedFMeasure
    val precision = trainingSummary.weightedPrecision
    val recall = trainingSummary.weightedRecall
    println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
      s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")

    //val accuracy = evaluator.evaluate(predicitveModel)
    //val mne = rModel.coefficientMatrix

   // model.save("D:/data/LogisticRegression")


  }

}