from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# Define schema
schema = "target INT, ids STRING, date STRING, flag STRING, user STRING, text STRING"

# Load data
def load_data(file_path):
    return spark \
        .read \
        .option("header", "false") \
        .schema(schema) \
        .csv(file_path)

# Define text preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text

# Register UDF
preprocess_udf = udf(preprocess_text, StringType())

def preprocess_data(df):
    return df.withColumn("cleaned_text", preprocess_udf(col("text")))

# Test with a local CSV file
df = load_data("C:/Users/aidag/Documents/PythonProj/SentimentAnalysisPyTorchSpark/SentimentAnalysisPyTorchSpark/sentiment140/training.1600000.processed.noemoticon.csv")
processed_df = preprocess_data(df)

# Show the first few rows of the processed DataFrame
processed_df.show(5)
