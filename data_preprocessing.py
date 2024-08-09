from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re

# Define schema
schema = "target INT, ids STRING, date STRING, flag STRING, user STRING, text STRING"

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

def load_data(spark, file_path):
    return spark \
        .readStream \
        .option("header", "false") \
        .schema(schema) \
        .csv(file_path)

# For real streamers like Kafka it would look like this:
# def load_data(spark, brokers, topic):
#     return spark \
#         .readStream \
#         .format("kafka") \
#         .option("kafka.bootstrap.servers", brokers) \
#         .option("subscribe", topic) \
#         .load()
