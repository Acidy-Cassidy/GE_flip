import requests
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. API Data Fetching
def get_cryptonews_general():
    url = "https://cryptonews-api.com/api/v1/category?section=general&items=100&page=1&token=it8y94a1ocabhpkodh3cxo5kny3sxqvq69gkmzqk"
    print(f"Fetching data from URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code}, Response: {response.text}")
        return "Error: " + str(response.status_code)

# 2. Data Preprocessing
def preprocess_data(data):
    processed_data = []
    dates = []
    sentiments = []
    for item in data.get('data', []):
        date_str = item['date']
        date_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = date_obj.strftime("%Y-%m-%d")
        sentiments.append(item['sentiment'])
        processed_item = {
            'title': item['title'],
            'text': item['text'],
            'topics': item['topics'],
            'sentiment': item['sentiment'],
            'date': formatted_date
        }
        processed_data.append(processed_item)
        dates.append(formatted_date)
    return processed_data, dates, sentiments

# 3. Text Processing and Tokenization
def tokenize_text(data):
    texts = [item['text'] for item in data]
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=200)
    return padded, tokenizer

# 4. Building the RNN Model
def build_rnn_model(input_length, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=512, input_length=input_length))
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Fetch Ethereum Sentiment Data
def get_ethereum_sentiment():
    url = "https://cryptonews-api.com/api/v1/stat?&section=general&items=200&page=1&token=it8y94a1ocabhpkodh3cxo5kny3sxqvq69gkmzqk"
    print(f"Fetching Ethereum sentiment data from URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch Ethereum sentiment data. Status Code: {response.status_code}, Response: {response.text}")
        return "Error: " + str(response.status_code)

def process_ethereum_sentiment(data):
    processed_data = {}
    if 'data' in data:
        for date, sentiment_info in data['data'].items():
            if isinstance(sentiment_info, dict):
                processed_data[date] = {
                    'neutral': sentiment_info.get('Neutral', 0),
                    'positive': sentiment_info.get('Positive', 0),
                    'negative': sentiment_info.get('Negative', 0),
                    'sentiment_score': sentiment_info.get('sentiment_score', 0)
                }
            else:
                print(f"Unexpected format for sentiment data on {date}: {sentiment_info}")
    else:
        print("No 'data' field in the API response.")
    return processed_data

def compare_sentiments(aggregated_predictions, eth_sentiment_data):
    print("\nComparison of Model Predictions with Ethereum Sentiment Data:")
    for date, sentiments in aggregated_predictions.items():
        eth_sentiments = eth_sentiment_data.get(date, {})
        print(f"Date: {date}")
        print(f"Model Predictions - Positive: {sentiments.get('positive', 0)}, Negative: {sentiments.get('negative', 0)}, Neutral: {sentiments.get('neutral', 0)}")
        if date in eth_sentiment_data:
            print(f"Ethereum Data - Positive: {eth_sentiments.get('positive', 'N/A')}, Negative: {eth_sentiments.get('negative', 'N/A')}, Neutral: {eth_sentiments.get('neutral', 'N/A')}")
        else:
            print("Ethereum Data - No data available for this date")
        print("")

# Main Execution
if __name__ == "__main__":
    raw_data = get_cryptonews_general()

    if isinstance(raw_data, str):
        print(raw_data)
    else:
        processed_data, dates, sentiments = preprocess_data(raw_data)

        X, tokenizer = tokenize_text(processed_data)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(sentiments)
        y = tf.keras.utils.to_categorical(encoded_labels)

        # Compute class weights for imbalanced classes
        class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels)
        class_weight_dict = dict(enumerate(class_weights))

        dates_series = pd.Series(dates)

        X_train, X_test, y_train, y_test, _, test_dates = train_test_split(X, y, dates_series, test_size=0.2, random_state=42, stratify=dates_series)
        model = build_rnn_model(X_train.shape[1], num_classes=y.shape[1])
        model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weight_dict)

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy*100:.2f}%")

        predictions = model.predict(X_test)
        predicted_sentiments = np.argmax(predictions, axis=1)
        unique_sentiments = label_encoder.classes_
        aggregated_predictions = {date: {sentiment.lower(): 0 for sentiment in unique_sentiments} for date in test_dates.unique()}

        for i, date in enumerate(test_dates):
            sentiment_label = label_encoder.inverse_transform([predicted_sentiments[i]])[0].lower()
            aggregated_predictions[date][sentiment_label] += 1

        print("Aggregated Predictions:")
        for date, sentiments in aggregated_predictions.items():
            print(f"Date: {date}, Positive: {sentiments.get('positive', 0)}, Negative: {sentiments.get('negative', 0)}, Neutral: {sentiments.get('neutral', 0)}")

        # Confusion Matrix and Classification Report
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
    
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=1))

        eth_sentiment_response = get_ethereum_sentiment()
        if isinstance(eth_sentiment_response, str):
            print(eth_sentiment_response)
        else:
            eth_sentiment_data = process_ethereum_sentiment(eth_sentiment_response)
            compare_sentiments(aggregated_predictions, eth_sentiment_data)
            # Print the number of articles processed
            print(f"Number of articles processed: {len(processed_data)}")
            # Print the number of sentiment scores used
            print(f"Number of sentiment scores used: {len(eth_sentiment_data)}")
