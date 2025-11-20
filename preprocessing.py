# preprocessing.py
import re
import nltk
from nltk.corpus import stopwords



# download resources if missing

try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
    

STOPWORDS=set(stopwords.words('english'))

url_pattern=re.compile(r'https?://\S+|www\.\S+')
mention_pattern=re.compile(r'@\w+')
hashtag_pattern=re.compile(r'#')
emoji_pattern=re.compile("["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
                         "]+",flags=re.UNICODE)


def clean_text(text:str, remove_stopwords:bool=True)->str:
    if not isinstance(text,str):
        return ""
    
    text=url_pattern.sub('',text)
    text=mention_pattern.sub('',text)
    text=hashtag_pattern.sub('',text)
    text=emoji_pattern.sub('',text)
    text=re.sub(r'[^0-9A-Za-z \t]+','',text)
    text=re.sub(r'\s+', ' ' ,text).strip().lower()
    
    if remove_stopwords:
        tokens=[t for t in text.split() if t not in STOPWORDS]
        return ' '.join(tokens)
    return text
