from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re

# Initialize Spark session
def create_spark_session():
    return SparkSession.builder \
        .appName("Sentiment Analysis") \
        .getOrCreate()

# Define schema
schema = "target INT, ids STRING, date STRING, flag STRING, user STRING, text STRING"

# Define text preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "otheruser", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text

# Register UDF
preprocess_udf = udf(preprocess_text, StringType())

# Function to preprocess and load data
def load_and_preprocess_data(spark, file_path):
    df = spark.read.option("header", "true").schema(schema).csv(file_path)
    df = df.withColumn("cleaned_text", preprocess_udf(col("text")))
    return df