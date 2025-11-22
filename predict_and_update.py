# predict_and_update.py

import pandas as pd
from database import get_conn
from model import VADERModel, DistilBERTModel

def fetch_unpredicted(limit=150):
    conn = get_conn()
    q = """
    SELECT tweet_id, content, clean_text
    FROM tweets
    WHERE vader_label IS NULL OR distil_label IS NULL
    ORDER BY datetime(inserted_at) DESC
    LIMIT ?
    """
    df = pd.read_sql_query(q, conn, params=(limit,))
    conn.close()
    return df

def update_predictions(df_preds: pd.DataFrame):
    conn = get_conn()
    cur = conn.cursor()

    for _, r in df_preds.iterrows():
        cur.execute("""
            UPDATE tweets
            SET vader_label=?, vader_score=?, distil_label=?, distil_score=?
            WHERE tweet_id=?
        """, (r['vader_label'], r['vader_score'], r['distil_label'], r['distil_score'], r['tweet_id']))

    conn.commit()
    conn.close()

def run_predictions(batch_size=32):
    vader = VADERModel()
    distil = DistilBERTModel()

    df = fetch_unpredicted(limit=150)
    if df.empty:
        print("No unpredicted tweets.")
        return

    texts = df['clean_text'].astype(str).tolist()

    vader_out = vader.predict(texts)

    distil_out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        distil_out.extend(distil.predict(batch))

    rows = []
    for i, tweet_id in enumerate(df['tweet_id']):
        rows.append({
            'tweet_id': tweet_id,
            'vader_label': vader_out[i]['label'],
            'vader_score': vader_out[i]['score'],
            'distil_label': distil_out[i]['label'],
            'distil_score': distil_out[i]['score']
        })

    update_predictions(pd.DataFrame(rows))
    print(f"Updated {len(rows)} tweets.")
