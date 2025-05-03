
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the Sentiment140 dataset (download from Kaggle first)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]

# Convert target labels (0 = negative, 4 = positive)
df['target'] = df['target'].map({0: 'negative', 4: 'positive'})

# Sample a smaller dataset for testing (optional)
df = df.sample(1000, random_state=42).reset_index(drop=True)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    return analyzer.polarity_scores(str(text))

# Apply sentiment analysis
df['vader_score'] = df['text'].apply(analyze_sentiment)

# Expand VADER scores into separate columns
df = pd.concat([df.drop(['vader_score'], axis=1), 
                df['vader_score'].apply(pd.Series)], axis=1)

# Map compound score to basic emotions
def map_emotion(compound):
    if compound >= 0.5:
        return 'joy'
    elif compound >= 0.1:
        return 'contentment'
    elif compound > -0.1:
        return 'neutral'
    elif compound > -0.5:
        return 'sadness'
    else:
        return 'anger'

df['emotion'] = df['compound'].apply(map_emotion)

# Plot emotion distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='viridis')
plt.title('Emotion Distribution from Tweets', fontsize=14)
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Export to CSV
df[['text', 'emotion']].to_csv('emotion_results.csv', index=False)

print("Done! Emotions have been classified and exported.")
