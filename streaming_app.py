from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from sentiment_model import predict_sentiment
from data_preprocessing import preprocess_streaming_data, load_data

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis Streaming") \
    .getOrCreate()
path_to_streaming_data = "C:/Users/aidag/Documents/PythonProj/SentimentAnalysisPyTorchSpark/SentimentAnalysisPyTorchSpark/sentiment140"
# Load the streaming data using the imported function
stream_df = load_data(spark, path_to_streaming_data)  # Replace with your actual data source

# Define a function to apply sentiment analysis
def apply_sentiment_analysis(df):
    def predict(text):
        # This will call the `predict_sentiment` function from your model
        return predict_sentiment(text)
    
    # Register UDF for prediction
    predict_udf = udf(predict, IntegerType())
    
    # Apply preprocessing and prediction
    df = preprocess_streaming_data(df)
    return df.withColumn("predicted_class", predict_udf(col("cleaned_text")))

# Apply sentiment analysis
processed_df = apply_sentiment_analysis(stream_df)

# Write the results to the console (for debugging/testing)
query = processed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
