# Telegram App Reviews – Sentiment Analysis (ML + NLP)

- This project performs full sentiment analysis on Telegram application reviews using:

    - Text preprocessing (regex cleaning, stopword removal, lemmatization)

    - TF-IDF vectorization

    - ML classifiers: Naive Bayes and Logistic Regression

    - Lexicon-based VADER sentiment scoring

    - Wordcloud visualization

    - Error analysis & feature importance

- The dataset has 3 sentiment labels:
    - Scores: 1-2 -> Label: Negative(0)
    - Score: 3 -> Label: Neutral(1)
    - Scores: 4-5 -> Label: Positive(2)

## Project Goals

1. Clean and preprocess Telegram review text

2. Split dataset into train/validation/test

3. Build ML models

4. Evaluate prediction performance

5. Identify the most important sentiment words

6. Visualize positive/negative wordclouds

7. Analyze typical model errors

## NLTK data

```
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
```

## Error Analysis Summary

- Common sources of mistakes:

    - Negation removed during preprocessing

    - Mixed emotional tone in one review

    - Sarcasm / implicit meaning

    - Domain-specific phrases (Telegram terminology)

    - Neutral class ambiguity

- Proposed improvements:

    - Keep negation words (“not”, “never”)

    - Domain-specific sentiment lexicon

    - Oversampling neutral examples

    - Use contextual models (BERT, RoBERTa)

## Wordcloud Examples

![](/assets/wordcloud.jpg)

- Top positive words: ['best' 'amazing' 'thank' 'great' 'love' 'perfect' 'wonderful' 'excellent'
 'useful' 'thanks' 'good' 'awesome' 'easy' 'status' 'messenger']
 
- Top negative words: ['worst' 'connecting' 'useless' 'bad' 'pathetic' 'even' 'update' 'code'
 'anymore' 'stupid' 'irritating' 'disappointed' 'suck' 'install'
 'internet']