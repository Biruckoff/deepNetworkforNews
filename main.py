from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)
import pandas as pd
from navec import Navec
from slovnet.model.emb import NavecEmbedding
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow import keras
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from keras.layers.merge import concatenate
from keras import backend as K
import csv
path = 'navec.tar'
tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)
navec = Navec.load(path)
tf.random.set_seed(1)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
data = pd.read_csv ("train.tsv", sep = '\t')
data1 = pd.read_csv ("predictions.tsv", sep = '\t')

size_data = data['title'].size
size_data1 = data1['title'].size

#Get vector from word
def getVect(word):
    vec = navec[word]
    return vec

#Get vector of emotions from word
def getEmotions(text):
    vec = np.zeros((5,))
    results = model.predict(text)
    sentiment_list = []
    for sentiment in results:
        sentiment_list.append(sentiment)
    for sentiment in sentiment_list:
            vec[0] = sentiment.get('negative')
            vec[1] = sentiment.get('skip')
            vec[2] = sentiment.get('neutral')
            vec[3] = sentiment.get('speech')
            vec[4] = sentiment.get('positive')
    return vec

#Syntactic and mophological analysis, get word's lemma
def processDoc(doc):
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        span.normalize(morph_vocab)
    return doc

#Bild the model
def getModel():
    layer1 = Input(shape=(300,), name = 'input1')
    hidden0 = Dense(150, activation='relu')(layer1)
    layer2 = Input(shape=(5,),name = 'input2')
    hidden6 = Dense(3, activation='relu')(layer2)
    merge = concatenate([hidden0, hidden6])
    hidden1 = Dense(100, activation='relu')(merge)
    hidden2 = Dense(75, activation='selu')(hidden1)
    hidden3 = Dense(50, activation='relu')(hidden2)
    hidden4 = Dense(20, activation='relu')(hidden3)
    out = Dense(1, activation='relu',name = 'output')(hidden4)
    model = keras.Model(inputs=[layer1, layer2],
                    outputs=out)
    return model

#F1-score    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

x_train = np.zeros((size_data,300))
x_train_e = np.zeros((size_data,5))
y_train = np.zeros((size_data,))

x_test = np.zeros((size_data1,300))
x_test_e = np.zeros((size_data1,5))
y_test = np.zeros((size_data1,1))

#get data from train.tsv
for i in range(size_data):
    doc = Doc(data['title'][i])
    doc = processDoc(doc)
   
    vec = np.zeros(300, )
    for token in doc.tokens:
        if not (token.rel == 'appos' or token.rel == 'flat:name' or len(token.text) < 3) and token.lemma in navec:
            vec = vec + getVect(token.lemma)
        
    
    emotion = getEmotions(data['title'][i])
    x_train_e[i] = emotion
    x_train[i] = vec
    y_train[i] = data['is_fake'][i]
    
#get data from predictions.tsv
for i in range(size_data1):
    doc = Doc(data1['title'][i])
    doc = processDoc(doc)
    
    vec = np.zeros(300, )
    for token in doc.tokens:
        if not (token.rel == 'appos' or token.rel == 'flat:name' or len(token.text) < 3) and token.lemma in navec:
            vec = vec + getVect(token.lemma)
        

    emotion = getEmotions(data['title'][i])
    x_test_e[i] = emotion
    x_test[i] = vec
    
model = getModel()

x_val = x_train[:1000]
y_val = y_train[:1000]
x_val_e = x_train_e[:1000]
x_train = x_train[1000:]
y_train = y_train[1000:]
x_train_e = x_train_e[1000:]

model.compile(optimizer='RMSprop', loss ='mean_squared_error', metrics=[f1_m])
model.fit({'input1': x_train, 'input2': x_train_e},
          y_train,
          epochs=70,
          batch_size=40,
)

answ = []
results = model.evaluate({'input1': x_val, 'input2': x_val_e}, y_val, batch_size=40)
print("test loss, test f1_m:", results)
y_test = model.predict({'input1': x_test, 'input2': x_test_e})
for i in range(len(y_test)):
    if y_test[i]>0.5:
        answ.append(1)
    else:
        answ.append(0)

with open('predictions.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['title', 'is_fake'])
    for i in range(len(y_test)):
        tsv_writer.writerow([data1['title'][i], answ[i]])
    

model.save('model')
 
