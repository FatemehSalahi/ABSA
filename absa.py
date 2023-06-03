#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers[sentencepiece]')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification \
  .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# Load a traditional Sentiment Analysis model
sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                          tokenizer=sentiment_model_path)
import pandas as pd
import re


# In[ ]:


df = pd.read_csv("/...path.../test.csv", encoding="utf-8", engine='python')
j=df.count(axis=0)[1]
dfabsaco=[]
pdabsa=[]


# In[ ]:


def absa(sent,aspect1):
    sentence=str(sent)
    aspect = aspect1
    inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    return probs


# In[ ]:


score=[0,0,0]
dfabsaco=[]
pdabsa=[]
a="..aspect.."
for i in range(j):
  sent=str(df['text'].iloc[i])
  text_word=sent.split(" ")
  for w in text_word:
    if w==a:
      score[0:2]=absa(sent,a)
      dfabsaco.append([i,sent,score[0],score[1],score[2]])
      break 
pdabsa=pd.DataFrame(dfabsaco,columns=['I','text','absaneg','absaneu','absapos'])
from google.colab import files
pdabsa.to_csv('/...path.../absatest.csv',encoding="utf-8")

