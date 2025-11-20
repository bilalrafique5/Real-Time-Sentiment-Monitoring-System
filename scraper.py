# scraper
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timezone


def fetch_tweets(query, limit=200, since=None):
    """
    Fetch up to 'limit' tweets matching 'the query' using sncraper.
    - Query: twitter search query string (e.g., 'Pakistan economy')
    - limit: max tweets to fetch
    - since: ISO date string 'YYYY-MM-DD' to limit earliest tweets (optional)
    
    Returns pandas.DataFrame with columns: id, date, content, username, url
    """
    
    tweets =[]
    q=query
    if since: 
        q+= f"since:{since}"
    scraper=sntwitter.TwitterSearchScraper(q)
    for i,tweet in enumerate(scraper.get_items()):
        if i>=limit:
            break
        tweets.append({
            'id':tweet.id,
            'date':tweet.date.replace(tzinfo=timezone.utc).isoformat(),
            'content':tweet.content,
            'username':tweet.user.username,
            'url':f"https://twitter.com/{tweet.user.username}/status/{tweet.id}"
           
        })
    
    df=pd.DataFrame(tweets)
    return df


if __name__=='__main__':
    df=fetch_tweets('Pakistan economy', limit=10)
    print(df.head())