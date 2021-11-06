import tensorflow as tf
from bs4 import BeautifulSoup
import mojimoji
import unicodedata
import re

def clean_html(html,strip=False):
    # print(html)
    soup=BeautifulSoup(html,'html.parser')
    text=soup.get_text(strip=strip)

    text=text.lower()
    text=mojimoji.zen_to_han(text)

    text=unicodedata.normalize("NFKC",text)

    # 連続した数字を0で置換
    text = re.sub(r'\d+\.+\d+', '0', text)
    text = re.sub(r'\d+\,+\d+','0',text)
    text = re.sub(r'\d+','0',text)
    text = re.sub(r'\!+','!',text)
    text = re.sub(r'\?\?+','??',text)
    text = re.sub(r'~~+','~~',text)
    text = re.sub(r'、、+','、、',text)
    text = re.sub(r'。。+','。。',text)
    text = re.sub(r'\s\s+','\s',text)
    text = re.sub(r'\n\n+','\n',text)
    text = re.sub(r'\r\r+','\r',text)
    text = re.sub(r'\t\t+','\t',text)
    text = re.sub(r'・・+','・・',text)
    text = re.sub(r'\.\.+','..',text)
    return text

def lower_text(text):
    return text.lower()

def preprocess_dataset(texts):
    texts=[clean_html(text) for text in texts]

    return texts

