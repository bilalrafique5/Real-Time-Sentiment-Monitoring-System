# predict_and_update
import pandas as pd
from database import get_conn
from model import VADERModel,DistilBERTModel
from config import DB_PATH

def fetch_unpredicted(limit=150):
    conn=get_conn()
    q="""   
    SELECT tweet_id, content, clean_text FROM tweets
    WHERE vader_label IS NULL OR distil_label IS NULL
    ORDER BY datetime(inserted_at) DESC
    LIMIT ?
    """
    
    
    df=pd.read_sql_query(q,conn,params=(limit,))
    conn.close()
    return df


def update_predictions(df_preds:pd.DataFrame):
    conn=get_conn()
    cur=conn.cursor()
    for _, row in df_preds.iterrows():
        cur.execute(""" UPDATE tweets
        SET vader_label = ?, vader_score = ?, distil_label = ?, distil_score = ?
        WHERE tweet_id = ?
        """, (row['vader_label'], row['vader_score'], row['distil_label'], row['distil_score'], row['tweet_id']))
    conn.commit()
    conn.close()
    
    
def run_predictions(batch_size=64):
    vader=VADERModel()
    distil=DistilBERTModel()
    to_predict=fetch_unpredicted(limit=150)
    if to_predict.empty:
        print("No unpredicted tweets.")
        return
    
    texts=to_predict['clean_text'].astype(str).tolist()
    #vader
    vader_out=vader.predict(texts)
    
    distil_out=[]
    for i in range(0,len(texts),batch_size):
        chunk=texts[i:i+batch_size]
        distil_out.extend(distil.predict(chunk))
        
    rows=[]
    for i, tid in enumerate(to_predict['tweet_id'].tolist()):
        v=vader_out[i]
        d=distil_out[i]
        rows.append({
            'tweet_id': tid,
            'vader_label': v['label'],
            'vader_score': v['score'],
            'distil_label': d.get('label'),
            'distil_score': d.get('score')
        })
    df_upd=pd.DataFrame(rows)
    update_predictions(df_upd)
    print(f"Updated {len(df_upd)} rows with predictions.")
    
    
if __name__=='__main__':
    run_predictions()
    