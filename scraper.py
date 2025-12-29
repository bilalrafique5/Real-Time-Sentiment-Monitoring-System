# scraper.py

import tweepy
import pandas as pd
from datetime import datetime

from config import API_BEARER_TOKEN, MAX_TWEETS_PER_CALL
from database import get_cached_tweets, insert_tweets_df, exists_tweet
from preprocessing import clean_text

# Initialize X (Twitter) Client
client = tweepy.Client(
    bearer_token=API_BEARER_TOKEN,
    wait_on_rate_limit=True
)


def search_recent_tweets(api_query: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch recent tweets using X API.
    Uses Firestore cache if available.
    Stores tweets with consistent schema for sentiment prediction.
    """

    # 1️⃣ Try cache first
    cached = get_cached_tweets(api_query, limit=limit)
    if not cached.empty and len(cached) >= limit:
        print(f"Returning {len(cached)} cached tweets for '{api_query}'")
        return cached

    print(f"Fetching tweets from X API for '{api_query}' (limit={limit})")

    tweets = []
    fetched = 0
    next_token = None

    while fetched < limit:
        to_fetch = min(limit - fetched, MAX_TWEETS_PER_CALL)

        try:
            resp = client.search_recent_tweets(
                query=api_query,
                max_results=to_fetch,
                expansions=["author_id"],
                tweet_fields=["id", "text", "created_at", "author_id"],
                user_fields=["username"],
                next_token=next_token
            )
        except Exception as e:
            print("Tweepy error:", e)
            break

        if not resp or not resp.data:
            break

        # Map author_id -> username
        users = {u.id: u.username for u in resp.includes.get("users", [])}

        for t in resp.data:
            tweet_id = str(t.id)

            # Skip duplicates
            if exists_tweet(tweet_id):
                continue

            tweets.append({
                "tweet_id": tweet_id,
                "query": api_query,
                "date": t.created_at.isoformat() if t.created_at else datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),   # REQUIRED for prediction ordering
                "username": users.get(t.author_id, ""),
                "content": t.text,
                "clean_text": clean_text(t.text),

                # Sentiment placeholders (updated later)
                "vader_label": None,
                "vader_score": None,
                "distil_label": None,
                "distil_score": None
            })

            fetched += 1

            if fetched >= limit:
                break

        next_token = resp.meta.get("next_token")
        if not next_token:
            break

    # 2️⃣ Insert new tweets into Firestore
    if tweets:
        df = pd.DataFrame(tweets)
        insert_tweets_df(df)
        print(f"Inserted {len(df)} new tweets into Firestore")
        return df

    # 3️⃣ Fallback to cache
    return cached
