`# SentimentAnalysisPyTorchSpark  ## Download Data  To download the dataset run:  
```bash kaggle datasets download -d kazanova/sentiment140`

About the Dataset
-----------------

[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

### Context

This dataset contains 1,600,000 tweets extracted using the Twitter API. The tweets are annotated as follows:

*   **0** = Negative
*   **4** = Positive

This dataset can be used for sentiment analysis.

### Content

The dataset contains the following fields:

*   **target**: The polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
*   **ids**: The ID of the tweet (e.g., 2087)
*   **date**: The date of the tweet (e.g., Sat May 16 23:58:44 UTC 2009)
*   **flag**: The query (e.g., lyx). If there is no query, this value is NO\_QUERY.
*   **user**: The user who tweeted (e.g., robotickilldozr)
*   **text**: The text of the tweet (e.g., Lyx is cool)

