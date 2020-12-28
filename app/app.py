from flask import Flask, request, jsonify, render_template
import os
import sys
import emojis
from unidecode import unidecode
import string
import re
import ast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizer, CamembertForSequenceClassification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


app = Flask(__name__)


def clean(text):
    
    # lower text and remove accents
    text = unidecode(text.lower())

    # remove numbers, words starting with # or @, and 'twitterlink' words    
    text = re.sub(r'#\S+|@\S+|\d+|twitterlink', r'', text)
    
    # remove remaining punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # return text with single whitespaces
    return ' '.join(text.split())


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods = ['POST', 'GET'])
def predict():

    # read text from form
    text = request.form.get('name')
    print(text)

    # tokenize text
    token_ids = torch.tensor(tokenizer.encode(clean(text))).unsqueeze(0)
    embedding = embedder(token_ids)[0]
    cls_embedding = np.matrix(embedding.squeeze()[0].detach().numpy())
    print('\nText succesfully embedded.\n')

    # run prediction
    pred = classifier.predict(cls_embedding)
    top_emoji = emoji_list[np.argmax(pred)]
    top3_emojis = [emoji_list[k] for k in np.argpartition(pred[0], -3)[-3:].tolist()]
    top3_emojis = ' '.join(top3_emojis)
    top_proba = max(pred[0])
    top3_proba = [pred[0][k] for k in np.argpartition(pred[0], -3)[-3:].tolist()]
    print('\nPrediction succesfully run.\n')

    return render_template('index.html', top1=top_emoji, 
                                         top3=top3_emojis,
                                         top1proba=top_proba,
                                         top3proba=top3_proba)


if __name__ == "__main__":
    
    # load model
    print('Loading model...')
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    embedder = CamembertModel.from_pretrained('camembert-base')
    classifier = keras.models.load_model("../models/100k_sentences_averaged_embedding.h5")
    print('\nModel successfuly loaded!\n')

    # load emoji list
    with open('top100.txt') as f:
        emoji_list = ast.literal_eval(f.read())
    
    app.run(host="0.0.0.0", port=5000, debug=True)
    

