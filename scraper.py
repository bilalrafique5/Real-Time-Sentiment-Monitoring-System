#scraper.py

import tweepy
import pandas as pd
from config import API_BEARER_TOKEN,MAX_TWEETS_PER_CALL
from database import get_cached_tweets, insert_tweets_df, exists_tweet
from preprocessing import clean_text
from datetime import datetime
from typing import List


# initialize tweepy client
client=tweepy.Client(bearer_token=API_BEARER_TOKEN, wait_on_rate_limit=True)

def search_recent_tweets(api_query:str, limit: int=200) -> pd.DataFrame:
    """   
    Returns a DataFrame of tweets for query (cached if available).
    Caching policy: will return cached tweets (from DB) if they exist and are recent (CACHE_HOURS).
    Otherwise, fetch from X API (Tweepy) until 'limit' tweets or exhausted.
    
    """
    # check DB cache first
    cached=get_cached_tweets(api_query, limit=limit)
    if not cached.empty and len(cached)>=limit:
        print(f"Returning {len(cached)} cached tweets for '{api_query}'")
        return cached
    
    
    # else fetch from API
    
    print(f"Fetching tweets from X API for '{api_query}' (limit={limit})")
    tweets=[]
    next_token=None
    fetched=0
    while fetched<limit:
        to_fetch=min(limit - fetched, MAX_TWEETS_PER_CALL)
        try:
            resp=client.search_recent_tweets(
                query=api_query,
                max_results=to_fetch,
                expansions=['author_id'],
                tweet_fields=['id','text','created_at','author_id'],
                user_fields=['username'],
                next_token=next_token
            )
        except Exception as e:
            print("Tweepy error:",e)
            break
        
        if resp is None or (resp.data is None):
            break
        
        users=[]
        if resp.includes and 'users' in resp.includes:
            for u in resp.includes['users']:
                users[u.id]=u.username
                
        for t in resp.data:
            tweet_id=str(t.id)
            if exists_tweet(tweet_id):
                # skip duplicates already in DB
                continue
            
            username=users.get(t.author_id, '')
            content=t.text 
            date=t.created_at.isoformat() if t.created_at else datetime.utcnow().isoformat()
            clean=clean_text(content)
            tweets.append({
                'tweet_id':tweet_id,
                'query':api_query,
                'date':date,
                'username':username,
                'content':content,
                'clean_text':clean,
                'vader_label':None,
                'vader_score':None,
                'distil_label':None,
                'distil_score':None
            })
            
            fetched+=1
        
        # handle pgination token
        meta=getattr(resp,'meta',None)
        next_token=meta.get('next_token') if meta else None
        if not next_token:
            break
        
    if tweets:
        df=pd.DataFrame(tweets)
        insert_tweets_df(df)
        return df
    
    return cached


if __name__=='__main__':
    df=search_recent_tweets("Pakistan economy -is:retweet lang:en",limit=50)
    print(df.head())