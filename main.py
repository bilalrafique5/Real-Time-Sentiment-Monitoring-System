from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

# existing modules
from scraper import search_recent_tweets
from database import (
    init_db,
    get_cached_tweets,
    insert_tweets_df,
    exists_tweet,
    db,
    COLLECTION,
)
from predict_and_update import run_predictions

app = FastAPI(title="Real-Time Sentiment Monitoring API")

# initialize db
init_db()

# --------------------
# Pydantic Schemas
# --------------------
class Tweet(BaseModel):
    tweet_id: str
    query: str
    date: str
    username: str
    content: str
    clean_text: str
    vader_label: Optional[str] = None
    vader_score: Optional[float] = None
    distil_label: Optional[str] = None
    distil_score: Optional[float] = None


class FetchRequest(BaseModel):
    query: str
    limit: int = 100


# --------------------
# CRUD Endpoints
# --------------------

@app.post("/tweets/fetch", response_model=List[Tweet], tags=["CREATE"])
def fetch_tweets(payload: FetchRequest):
    """
    Fetch tweets from X API (or cache) and store them in Firestore.
    """
    df = search_recent_tweets(payload.query, payload.limit)
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


@app.get("/tweets", response_model=List[Tweet], tags=["READ"])
def read_tweets(
    query: str = Query(...),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Read cached tweets for a specific query.
    """
    df = get_cached_tweets(query, limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.get("/tweets/{tweet_id}", response_model=Tweet, tags=["READ"])
def read_single_tweet(tweet_id: str):
    """
    Get a single tweet by ID.
    """
    doc = db.collection(COLLECTION).document(tweet_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Tweet not found")
    data = doc.to_dict()
    data["tweet_id"] = tweet_id
    return data


@app.delete("/tweets/{tweet_id}", tags=["DELETE"])
def delete_tweet(tweet_id: str):
    """
    Delete a tweet by ID.
    """
    if not exists_tweet(tweet_id):
        raise HTTPException(status_code=404, detail="Tweet not found")
    db.collection(COLLECTION).document(tweet_id).delete()
    return {"message": "Tweet deleted successfully"}


@app.put("/tweets/{tweet_id}", tags=["UPDATE"])
def update_tweet_sentiment(
    tweet_id: str,
    vader_label: Optional[str] = None,
    vader_score: Optional[float] = None,
    distil_label: Optional[str] = None,
    distil_score: Optional[float] = None,
):
    """
    Update sentiment fields of a tweet.
    """
    if not exists_tweet(tweet_id):
        raise HTTPException(status_code=404, detail="Tweet not found")

    update_data = {}
    if vader_label is not None:
        update_data["vader_label"] = vader_label
    if vader_score is not None:
        update_data["vader_score"] = vader_score
    if distil_label is not None:
        update_data["distil_label"] = distil_label
    if distil_score is not None:
        update_data["distil_score"] = distil_score

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    db.collection(COLLECTION).document(tweet_id).update(update_data)
    return {"message": "Tweet updated successfully"}


# --------------------
# Prediction Endpoint
# --------------------

@app.post("/tweets/predict", tags=["PREDICT"])
def predict_unlabeled_tweets():
    """
    Run VADER and DistilBERT predictions on unlabeled tweets.
    """
    run_predictions()
    return {"message": "Predictions executed successfully"}


# --------------------
# Health Check
# --------------------

@app.get("/", tags=["READ"])
def root():
    return {"status": "API running"}
