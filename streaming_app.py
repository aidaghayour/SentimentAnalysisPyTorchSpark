import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from sentiment_model import predict_sentiment
from data_preprocessing import preprocess_streaming_data, load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis Streaming") \
    .getOrCreate()

logger.info("Spark session initialized.")

# Define the path to streaming data
path_to_streaming_data = "C:/Users/aidag/Documents/PythonProj/SentimentAnalysisPyTorchSpark/SentimentAnalysisPyTorchSpark/sentiment140"

# Check if the path exists
if not os.path.exists(path_to_streaming_data):
    raise FileNotFoundError(f"The path {path_to_streaming_data} does not exist.")

# Load the streaming data using the imported function
try:
    stream_df = load_data(spark, path_to_streaming_data)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Define a function to apply sentiment analysis
def apply_sentiment_analysis(df):
    def predict(text):
        # This will call the `predict_sentiment` function from your model
        return predict_sentiment(text)
    
    # Register UDF for prediction
    predict_udf = udf(predict, IntegerType())
    
    # Apply preprocessing and prediction
    logger.info("Preprocessing data and applying sentiment analysis...")
    df = preprocess_streaming_data(df)
    df = df.withColumn("predicted_class", predict_udf(col("cleaned_text")))
    
    logger.info("Sentiment analysis applied.")
    return df

# Apply sentiment analysis
try:
    processed_df = apply_sentiment_analysis(stream_df)
except Exception as e:
    print(f"Error applying sentiment analysis: {e}")
    raise

# Write the results to the console (for debugging/testing)
query = processed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

logger.info("Streaming started. Awaiting termination...")

# Wait for the streaming to finish
query.awaitTermination()
