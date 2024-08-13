from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import datetime
from pyspark.sql.functions import min

# Initialize Spark session
spark = SparkSession.builder \
        .appName("Sentiment Analysis") \
        .getOrCreate()

file_path = "./sentiment140/training.1600000.processed.noemoticon.csv"

# Define schema
schema = "target INT, ids STRING, date STRING, flag STRING, user STRING, text STRING"

df = spark.read.option("header", "false").schema(schema).csv(file_path)

# UDF to calculate the step
def batch_up(date_str):
    timestamp = datetime.datetime.strptime(date_str, "%a %b %d %H:%M:%S PDT %Y")
    epoch_time = (timestamp - datetime.datetime(1970, 1, 1)).total_seconds()
    step = int(epoch_time // (6 * 3600)) + 1  # 6 hours = 6 * 3600 seconds
    return step

# Register the UDF
batch_up_udf = F.udf(batch_up, T.IntegerType())


df_with_step = df.withColumn("step", batch_up_udf(F.col("date")))
df_with_step.show()

df_with_step.groupBy("step").count().show(10)


# now we save each batch of time, into a seprate file (Long process!)
# Get distinct step values
steps = df_with_step.select("step").distinct().collect()

# for step in steps[:]:
#     print(step)
#     print(step[0])
#     print(df_with_step["step"])
#     _df = df_with_step.filter(df_with_step["step"] == step[0])
#     _df.show(10)


#     print("start writing")
#     _df.coalesce(1).write.mode("append").option("header","true").csv("data/steps")
#     print("Done writing")

part = spark.read.csv("data/steps/part-00000-017f1107-fc07-470f-9703-f59008832742-c000.csv", header=True, inferSchema=True)
part.groupBy("step").count().show()

dataSchema = part.schema
print(dataSchema)

streaming = spark.readStream.schema(dataSchema).option("maxFilesPerTrigger",1).csv("data/steps")