from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

import re
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()


# Define schema
schema = "target INT, ids STRING, date STRING, flag STRING, user STRING, text STRING"

# Load data
def load_data(file_path):
    return spark \
        .readStream \
        .option("header", "false") \
        .schema(schema) \
        .csv(file_path)


# for real streamers like Kafka it would look like this:
# def load_data(brokers, topic):
#     return spark \
#         .readStream \
#         .format("kafka") \
#         .option("kafka.bootstrap.servers", brokers) \
#         .option("subscribe", topic) \
#         .load()

# Define text preprocessing function
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Replace @username with "otheruser"
    text = re.sub(r"@\w+", "otheruser", text)
    # Remove special characters and punctuations
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

# Register UDF
preprocess_udf = udf(preprocess_text, StringType())

def preprocess_streaming_data(stream_df):
    return stream_df.withColumn("cleaned_text", preprocess_udf(col("text")))