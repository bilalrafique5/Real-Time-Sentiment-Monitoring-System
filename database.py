import firebase_admin
from google.cloud.firestore_v1 import FieldFilter
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta
from config import CACHE_HOURS

# Initialize Firebase only once
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

COLLECTION = "tweets"

def init_db():
    # Firestore is schemaless, no table creation needed
    pass

def insert_tweets_df(df: pd.DataFrame):
    """Insert a DataFrame of tweets into Firestore."""
    if df is None or df.empty:
        return
    for _, row in df.iterrows():
        doc_ref = db.collection(COLLECTION).document(row['tweet_id'])
        doc_ref.set({
            'query': row['query'],
            'date': row['date'],
            'username': row['username'],
            'content': row['content'],
            'clean_text': row['clean_text'],
            'vader_label': row['vader_label'],
            'vader_score': row['vader_score'],
            'distil_label': row['distil_label'],
            'distil_score': row['distil_score'],
            'inserted_at': datetime.utcnow().isoformat()
        })

def get_cached_tweets(query: str, limit: int = 100):
    """Return cached tweets for a query within CACHE_HOURS."""
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_HOURS)
    docs = (
        db.collection(COLLECTION)
        .where(filter=FieldFilter("query", "==", query))
        .where(filter=FieldFilter("inserted_at", ">=", cutoff.isoformat()))
        .order_by("inserted_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
)


    rows = []
    for doc in docs:
        d = doc.to_dict()
        d['tweet_id'] = doc.id
        rows.append(d)

    if not rows:
        return pd.DataFrame(columns=[
            'tweet_id', 'query', 'date', 'username', 'content',
            'clean_text', 'vader_label', 'vader_score',
            'distil_label', 'distil_score', 'inserted_at'
        ])
    return pd.DataFrame(rows)

def exists_tweet(tweet_id: str) -> bool:
    """Check if a tweet already exists in Firestore."""
    doc_ref = db.collection(COLLECTION).document(tweet_id)
    return doc_ref.get().exists
