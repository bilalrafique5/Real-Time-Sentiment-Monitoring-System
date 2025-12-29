import pandas as pd
from database import db, COLLECTION
from model import VADERModel, DistilBERTModel

def fetch_unpredicted(limit=150):
    docs = (
        db.collection(COLLECTION)
        .where("vader_label", "==", None)
        .where("distil_label", "==", None)
        .limit(limit)
        .stream()
    )

    rows = []
    for doc in docs:
        d = doc.to_dict()
        d["tweet_id"] = doc.id
        rows.append(d)

    return pd.DataFrame(rows)



def update_predictions(df_preds: pd.DataFrame):
    for _, r in df_preds.iterrows():
        db.collection(COLLECTION).document(r['tweet_id']).update({
            'vader_label': r['vader_label'],
            'vader_score': r['vader_score'],
            'distil_label': r['distil_label'],
            'distil_score': r['distil_score']
        })


def run_predictions(batch_size=32):
    """Run VADER and DistilBERT predictions on upredicted tweets."""
    
    vader=VADERModel()
    distil=DistilBERTModel()
    
    df=fetch_unpredicted(limit=150)
    if df.empty:
        print("No unpredicted tweets")
        return
    
    texts=df['clean_text'].astype(str).tolist()
    
    vader_out=vader.predict(texts)
    distil_out=[]
    for i in range(0, len(texts), batch_size):
        batch=texts[i:i+batch_size]
        distil_out.extend(distil.predict(batch))
    
    rows=[]
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