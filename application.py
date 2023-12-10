import numpy as np
from flask import Flask, request, jsonify, render_template
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, lower

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.master("local").appName("MyApp").getOrCreate()

# Load the trained model
MODEL = PipelineModel.load("/Users/csuftitan/Documents/Spark/model_cv_nb")

HTTP_BAD_REQUEST = 400

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    Description = request.args.get('Description', default=None, type=str)
    #Description = request.form.get('Description')
    # Reject requests that have bad or missing values.
    if Description is None:
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because of '
                   'missing or unacceptable values. '
                   'All values must be present and of type string.')
        response = jsonify(status='error', error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response
    
    # Create a DataFrame with the given Description
    data = [(Description,)]
    schema = ["Description"]
    df = spark.createDataFrame(data, schema=schema)

    # Apply tokenization (or any other necessary preprocessing)
    tokenizer = Tokenizer(inputCol="Description", outputCol="tokens")
    df = tokenizer.transform(df)

    # Apply stop words removal
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_words")
    df = remover.transform(df)

    # Use CountVectorizer to convert the words into a numerical representation
    vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")
    model = vectorizer.fit(df)  # Fit the vectorizer on the training data
    df = model.transform(df)    # Transform the DataFrame using the fitted vectorizer

    # Drop the "tokens" column to avoid conflicts with loaded model
    df = df.drop("tokens")
    df = df.drop("filtered_words")
    df = df.drop("features")

    #result_df = spark.read.format('csv')\
          #.option('header','true')\
          #.option('inferSchema', 'true')\
          #.option('timestamp', 'true')\
          #.load('hdfs://127.0.0.1:9000/user/ADB_Project/train.csv')

    #result_df = result_df.groupBy('Category').count() \
    #.withColumnRenamed('count', 'totalValue') \
    #.orderBy(col('totalValue').desc())

    #first_column_values = [row['Category'] for row in result_df.collect()]
    first_column_values=['LARCENY/THEFT', 'OTHER OFFENSES', 'NON-CRIMINAL', 'ASSAULT', 'DRUG/NARCOTIC', 'VEHICLE THEFT', 'VANDALISM', 'WARRANTS', 'BURGLARY', 'SUSPICIOUS OCC', 'MISSING PERSON', 'ROBBERY', 'FRAUD', 'FORGERY/COUNTERFEITING', 'SECONDARY CODES', 'WEAPON LAWS', 'PROSTITUTION', 'TRESPASS', 'STOLEN PROPERTY', 'SEX OFFENSES FORCIBLE', 'DISORDERLY CONDUCT', 'DRUNKENNESS', 'RECOVERED VEHICLE', 'KIDNAPPING', 'DRIVING UNDER THE INFLUENCE', 'RUNAWAY', 'LIQUOR LAWS', 'ARSON', 'LOITERING', 'EMBEZZLEMENT', 'SUICIDE', 'FAMILY OFFENSES', 'BAD CHECKS', 'BRIBERY', 'EXTORTION', 'SEX OFFENSES NON FORCIBLE', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']
    # Load the pre-trained model
    loaded_model = PipelineModel.load("/Users/csuftitan/Documents/Spark/model_cv_nb")

    # Use the model to transform the DataFrame
    result = loaded_model.transform(df)

    # Show the result
    result.select("Description", "features", "prediction", "probability").show(truncate=False)

    # Evaluate the prediction
    #evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="probability", metricName="accuracy")
    #evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="rawPrediction", metricName="accuracy")
    #accuracy = evaluator.evaluate(result)
    print(first_column_values[int(result.select("prediction").first()[0])])
    return render_template('index.html', prediction_text='Pridicted Crime Category is {}'.format(first_column_values[int(result.select("prediction").first()[0])]))
    return jsonify(status='complete', prediction=result.select("prediction").first()[0], text=first_column_values[int(result.select("prediction").first()[0])])

if __name__ == "__main__":
    app.run(debug=True)
