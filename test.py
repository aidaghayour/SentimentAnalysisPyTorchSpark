from data_preprocessing import load_data, preprocess_streaming_data  # Import your functions

# Load data (replace with your actual path)
df = load_data("C:/Users/aidag/Documents/PythonProj/SentimentAnalysisPyTorchSpark/SentimentAnalysisPyTorchSpark/sentiment140/training.1600000.processed.noemoticon.csv")

# Apply preprocessing
processed_df = preprocess_streaming_data(df)

# Display the result (for testing purposes)
query = processed_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

query.awaitTermination()
