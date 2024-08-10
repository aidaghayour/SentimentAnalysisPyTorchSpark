import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)
# Load the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)


# Function to preprocess input and make predictions
def predict_sentiment(text):
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, clean_up_tokenization_spaces=True)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class