import pandas as pd
from data_preprocessing import *
from sentiment_model import predict_sentiment


def main():
    # Set up Spark session
    spark = create_spark_session()

    # Path to your data file
    file_path = "C:/Users/aidag/Documents/PythonProj/SentimentAnalysisPyTorchSpark/SentimentAnalysisPyTorchSpark/sample.csv"

    # Load and preprocess data
    df = load_and_preprocess_data(spark, file_path)

    # Convert DataFrame to Pandas for PyTorch
    pandas_df = df.select("cleaned_text").toPandas()
    texts = pandas_df["cleaned_text"].tolist()



    # Predict sentiment
    predictions = predict_sentiment(texts)
    print(predictions)

if __name__ == "__main__":
    main()