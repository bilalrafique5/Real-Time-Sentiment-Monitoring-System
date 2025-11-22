# streamlit_app.py
import streamlit as st
import pandas as pd
from scraper import search_recent_tweets
from database import init_db, get_cached_tweets, get_conn
from predict_and_update import run_predictions
from model import VADERModel, DistilBERTModel
from plotly import express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

init_db()
st.set_page_config(layout='wide', page_title='Real-time Sentiment Monitor')

st.title("Real-Time Sentiment Monitoring System (Formerly Twitter)")

with st.sidebar:
    query = st.text_input("Search query", value="Pakistan economy -is:retweet lang:en")
    limit = st.slider("Number of tweets to fetch", 10, 500, 100, 10)
    models = st.multiselect("Use models", ["VADER", "DistilBERT"], default=["VADER", "DistilBERT"])
    fetch_btn = st.button("Fetch (uses cache when possible)")
    predict_btn = st.button("Run predictions on uncategorized tweets")
    refresh = st.button("Refresh view")

if fetch_btn:
    with st.spinner("Fetching tweets (cached when possible)..."):
        df = search_recent_tweets(query, limit=limit)
        st.success(f"Fetched & cached {len(df)} tweets (new).")

# show latest tweets from DB for this query
df_db = get_cached_tweets(query, limit=limit)
if df_db.empty:
    st.info("No cached tweets for this query yet. Click Fetch.")
else:
    st.subheader("Tweet feed (latest cached)")
    for _, r in df_db.head(50).iterrows():
        st.markdown(f"**@{r['username']}** — {r['date']}")
        st.write(r['content'])
        cols = st.columns(2)
        cols[0].write(f"VADER: {r['vader_label']}, {r['vader_score']}")
        cols[1].write(f"Distil: {r['distil_label']}, {r['distil_score']}")
        st.markdown("---")

    # Summary stats
    st.subheader("Summary")
    if 'vader_label' in df_db.columns:
        vader_dist = df_db['vader_label'].value_counts(normalize=True).mul(100).round(2).to_dict()
        st.write("VADER distribution %:", vader_dist)
    if 'distil_label' in df_db.columns:
        distil_dist = df_db['distil_label'].value_counts(normalize=True).mul(100).round(2).to_dict()
        st.write("DistilBERT distribution %:", distil_dist)

    # time-series chart (VADER)
    try:
        df_db['dt'] = pd.to_datetime(df_db['date'])
        ts = df_db.groupby([pd.Grouper(key='dt', freq='1Min'), 'vader_label']).size().reset_index(name='count')
        fig = px.line(ts, x='dt', y='count', color='vader_label', title='VADER sentiment over time')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # wordcloud of positive (vader)
    pos_text = ' '.join(df_db[df_db['vader_label']=='POSITIVE']['clean_text'].astype(str).tolist())
    if pos_text:
        wc = WordCloud(width=800, height=300).generate(pos_text)
        plt.figure(figsize=(12,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

if predict_btn:
    with st.spinner("Running models... (may take a while for DistilBERT)"):
        run_predictions()
        st.success("Predictions complete. Refresh view to see updated labels.")
