# scraper.py

import tweepy
import pandas as pd
from config import API_BEARER_TOKEN, MAX_TWEETS_PER_CALL
from database import get_cached_tweets, insert_tweets_df, exists_tweet
from preprocessing import clean_text
from datetime import datetime

client = tweepy.Client(bearer_token=API_BEARER_TOKEN, wait_on_rate_limit=True)

def search_recent_tweets(api_query:str, limit:int=200)->pd.DataFrame:
    cached=get_cached_tweets(api_query, limit=limit)
    if not cached.empty and len(cached)>=limit:
        print(f"Returning {len(cached)} cached tweets for '{api_query}'")
        return cached
    
    print(f"Fetching tweets from X API for '{api_query}' (limit={limit})")
    tweets=[]
    next_token=None
    fetched=0
    
    while fetched<limit:
        to_fetch=min(limit-fetched, MAX_TWEETS_PER_CALL)
        try:
            resp=client.search_recent_tweets(
                query=api_query,
                max_results=to_fetch,
                expansions=['author_id'],
                tweet_fields=['id', 'text', 'created_at', 'author_id'],
                user_fields=['username'],
                next_token=next_token
            )
        except Exception as e:
            print("Tweepy error:",e)
            break
        
        if not resp or not resp.data:
            break
        
        users={u.id:u.username for u in resp.includes.get("users",[])}
        
        for t in resp.data:
            tid=str(t.id)
            if exists_tweet(tid):
                continue
            
            tweets.append({
                'tweet_id':tid,
                'query':api_query,
                'date':t.created_at.isoformat() if t.created_at else datetime.utcnow().isoformat(),
                'username': users.get(t.author_id, ''),
                'content': t.text,
                'clean_text': clean_text(t.text),
                'vader_label': None,
                'vader_score': None,
                'distil_label': None,
                'distil_score': None
            })
            
            fetched += 1

        next_token = resp.meta.get("next_token")
        if not next_token:
            break

    if tweets:
        df = pd.DataFrame(tweets)
        insert_tweets_df(df)  # now inserts into Firestore
        return df

    return cached
                
