# model.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

class VADERModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, texts):
        out = []
        for t in texts:
            vs = self.analyzer.polarity_scores(t)
            comp = vs['compound']

            if comp >= 0.05:
                label = 'POSITIVE'
            elif comp <= -0.05:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'

            out.append({'label': label, 'score': comp, 'raw': vs})
        return out


class DistilBERTModel:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        self.pipe = pipeline('sentiment-analysis', model=model_name)

    def predict(self, texts, batch_size=16):
        return self.pipe(texts, truncation=True)
