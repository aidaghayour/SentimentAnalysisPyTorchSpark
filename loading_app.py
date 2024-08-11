import pandas as pd
from data_preprocessing import *
from sentiment_model import predict_sentiment
from plotting import update_plot



def main():
    # Set up Spark session
    spark = create_spark_session()

    # Path to your data file
    file_path = "C:/Users/aidag/Documents/PythonProj/SentimentAnalysisPyTorchSpark/SentimentAnalysisPyTorchSpark/sentiment140/training.1600000.processed.noemoticon.csv"

    # Load and preprocess data
    df = load_and_preprocess_data(spark, file_path)

    # Convert DataFrame to Pandas for PyTorch
    pandas_df = df.select("cleaned_text", "date").toPandas()
    texts = pandas_df["cleaned_text"].tolist()
    timestamps = pandas_df["date"].tolist()  # Extract timestampsfor plotting



    # Predict sentiment
    predictions = predict_sentiment(texts)
    print(predictions)

    # Update plot
    update_plot(timestamps, predictions)

if __name__ == "__main__":
    main()