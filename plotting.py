import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def update_plot(timestamps, predictions):
    # Convert timestamps to datetime objects
    timestamps = pd.to_datetime(timestamps)

    # Create a DataFrame with timestamps and predictions
    df = pd.DataFrame({"timestamp": timestamps, "sentiment": predictions})

    # Group by hour of the day and sentiment
    df['hour'] = df['timestamp'].dt.hour
    hourly_sentiment = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)

    # Plot the sentiments
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_sentiment.index, hourly_sentiment.get(1, 0), color='green', label='Positive Sentiment')
    plt.plot(hourly_sentiment.index, hourly_sentiment.get(0, 0), color='red', label='Negative Sentiment')

    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Sentiments')
    plt.title('Sentiment Distribution Over the Day')
    plt.legend()
    plt.grid(True)
    plt.show()
