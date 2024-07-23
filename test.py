import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns

# Load your dataset
df = pd.read_csv('twitter_training.csv')

# Display the first few rows and column names to understand the data
print(df.head())
print(df.columns)

# Define the text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ''
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (optional, if not needed for analysis)
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

# Replace 'text_column_name' with the actual name of your text column
df['cleaned_text'] = df['Feedback'].apply(clean_text)

# Check the results
print(df[['Feedback', 'cleaned_text']].head())

# Save the cleaned dataset
df.to_csv('cleaned_twitter_data.csv', index=False)


text = ' '.join(df['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Define a function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(text)
    # Return sentiment polarity (positive, negative, neutral)
    return analysis.sentiment.polarity

# Apply the function to the cleaned text column
df['sentiment_polarity'] = df['cleaned_text'].apply(get_sentiment)

# Print a sample of the DataFrame with sentiment
print(df[['cleaned_text', 'sentiment_polarity']].head())


# Plot distribution of sentiment polarity
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_polarity'], kde=True, bins=30)
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

df.to_csv('sentiment_analysis_results.csv', index=False)