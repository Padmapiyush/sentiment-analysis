# sentiment-analysis
Conducted sentiment analysis on customer reviews using natural language processing techniques in Python.


Here is an example code for performing sentiment analysis on customer reviews using natural language processing techniques in Python. This code uses the NLTK library for text preprocessing and the TextBlob library for sentiment analysis.

```python
# Imported necessary libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Loaded the customer review dataset
df = pd.read_csv('customer_reviews.csv')

# Removed punctuation and lowercase all text
df['clean_text'] = df['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha()]))

# Removed stop words
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if not word in stop_words]))

# Performed sentiment analysis using TextBlob
df['sentiment'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Assigned a label based on sentiment score
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Outputed the results
print(df[['text', 'sentiment_label']]) 
```

In this code, we first load the customer review dataset and preprocess the text by removing punctuation and stop words. We then perform sentiment analysis using TextBlob and assign a label (positive, negative, or neutral) based on the sentiment score. Finally, we output the results, which include the original text and the sentiment label.

Note that this code is just an example and may need to be modified to suit your specific use case. For example, you may need to use a different dataset or adjust the text preprocessing steps to better fit your data.
