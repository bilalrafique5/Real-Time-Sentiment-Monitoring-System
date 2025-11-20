# database.py

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from config import DB_PATH, CACHE_HOURS


CREATE_SQL="""
CREATE TABLE IF NOT EXISTS tweets(
    tweet_id text PRIMARY KEY,
    query text,
    date text,
    username text,
    content text,
    clean_text text,
    vader_label text,
    vader_score REAL,
    distil_label text,
    distil_score REAL,
    inserted_at DATETIME DEFAULT (datetime('now'))
    
    
    
    
); 

"""
def get_conn():
    conn=sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn=get_conn()
    cur=conn.cursor()
    cur.execute(CREATE_SQL)
    conn.commit()
    conn.close()
    
def insert_tweets_df(df:pd.DataFrame):
    if df is None or df.empty:
        return
    conn=get_conn()
    df.to_sql('tweets',conn,if_exists='append',index=False)
    conn.commit()
    conn.close()
    
    
def get_cached_tweets(query:str, limit: int=100):
    """ 
    return cached tweets for this query if they were insereted within CACHE_HOURS
    
    """
    conn=get_conn()
    cutoff=datetime.utcnow() -  timedelta(hours=CACHE_HOURS)
    cutoff_str=cutoff.strftime("%Y-%m-%d %H:%M:%S")
    q=""" 
    SELECT tweet_id, query, date, username, content, clean_text,vader_label,
              vader_score, distil_label, distil_score, inserted_at
    FROM tweets
    WHERE query = ?
        AND datetime(inserted_at) >= datetime(?)
    ORDER BY datetime(data) DESC
    LIMIT ?
    
    """
    cur=conn.cursor()
    cur.execute(q, (query, cutoff_str, limit))
    rows= cur.fetchall()
    cols=[c[0] for c in cur.description]
    conn.close()
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)

def exists_tweet(tweet_id:str)->bool:
    conn=get_conn()
    cur=conn.cursor()
    cur.execute("SELECT 1 FROM tweets WHERE tweet_id=? LIMIT 1",(tweet_id,))
    r=cur.fetchone()
    conn.close()
    return r is  not None

