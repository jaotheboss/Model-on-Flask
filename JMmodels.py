"""
Creating a Flask API for Python backend

Execution:
    1. Run this script on Terminal
"""

import os
origin = '/Users/jaoming/Documents/Active Projects/Models on Flask'
os.chdir(origin)

# importing the relevant tools and modules
from flask import Flask
from flask import render_template # so that we can take in html templates and apply them # used to reference the layout template
from flask import url_for # to retrieve files in other directories # used to retrieve css template
from flask import request

# overall
import pandas as pd
import numpy as np

# for model 1
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
model = load_model('Model_2020-05-05.h5')

app = Flask(__name__)

# code for Model 1
col_names = ['Career', 'COVID-19', 'CPF', 'Credit Cards', 'Entrepreneurship', 
                            'Family', 'Insurance', 'Investments', 'Lifestyle', 'Loans', 'Payments', 
                            'Property', 'Retirement', 'Savings Accounts', 'Savings', 
                            'Stock Discussion', 'Bonds', 'Brokerages', 'Cryptocurrency',
                            'Dividends', 'ETF', 'Fresh Graduate', 'HDB BTO', 'Miles', 'REIT',
                            'Robo-Advisor', 'Unit Trust']

def _PREPROCESS_Q(questions, extent = 'full'):
       """
       Function:     Acts as a sub function to the bigger _PREPROCESS.
                     This function seeks to only preprocess the questions
                     
       Input:        Questions column
       
       Returns:      A column of reprocessed questions
       """
       ## for manipulating the questions
       stop_words = stopwords.words('english')
       stop_words.extend(['hi', 'hello', 'amp'])

       ps = PorterStemmer()

       contractions = {
              "ain't": "am not / are not",
              "aren't": "are not / am not",
              "can't": "cannot",
              "can't've": "cannot have",
              "'cause": "because",
              "could've": "could have",
              "couldn't": "could not",
              "couldn't've": "could not have",
              "didn't": "did not",
              "doesn't": "does not",
              "don't": "do not",
              "hadn't": "had not",
              "hadn't've": "had not have",
              "hasn't": "has not",
              "haven't": "have not",
              "he'd": "he had / he would",
              "he'd've": "he would have",
              "he'll": "he shall / he will",
              "he'll've": "he shall have / he will have",
              "he's": "he has / he is",
              "how'd": "how did",
              "how'd'y": "how do you",
              "how'll": "how will",
              "how's": "how has / how is",
              "i'd": "I had / I would",
              "i'd've": "I would have",
              "i'll": "I shall / I will",
              "i'll've": "I shall have / I will have",
              "i'm": "I am",
              "i've": "I have",
              "isn't": "is not",
              "it'd": "it had / it would",
              "it'd've": "it would have",
              "it'll": "it shall / it will",
              "it'll've": "it shall have / it will have",
              "it's": "it has / it is",
              "let's": "let us",
              "ma'am": "madam",
              "mayn't": "may not",
              "might've": "might have",
              "mightn't": "might not",
              "mightn't've": "might not have",
              "must've": "must have",
              "mustn't": "must not",
              "mustn't've": "must not have",
              "needn't": "need not",
              "needn't've": "need not have",
              "o'clock": "of the clock",
              "oughtn't": "ought not",
              "oughtn't've": "ought not have",
              "shan't": "shall not",
              "sha'n't": "shall not",
              "shan't've": "shall not have",
              "she'd": "she had / she would",
              "she'd've": "she would have",
              "she'll": "she shall / she will",
              "she'll've": "she shall have / she will have",
              "she's": "she has / she is",
              "should've": "should have",
              "shouldn't": "should not",
              "shouldn't've": "should not have",
              "so've": "so have",
              "so's": "so as / so is",
              "that'd": "that would / that had",
              "that'd've": "that would have",
              "that's": "that has / that is",
              "there'd": "there had / there would",
              "there'd've": "there would have",
              "there's": "there has / there is",
              "they'd": "they had / they would",
              "they'd've": "they would have",
              "they'll": "they shall / they will",
              "they'll've": "they shall have / they will have",
              "they're": "they are",
              "they've": "they have",
              "to've": "to have",
              "wasn't": "was not",
              "we'd": "we had / we would",
              "we'd've": "we would have",
              "we'll": "we will",
              "we'll've": "we will have",
              "we're": "we are",
              "we've": "we have",
              "weren't": "were not",
              "what'll": "what shall / what will",
              "what'll've": "what shall have / what will have",
              "what're": "what are",
              "what's": "what has / what is",
              "what've": "what have",
              "when's": "when has / when is",
              "when've": "when have",
              "where'd": "where did",
              "where's": "where has / where is",
              "where've": "where have",
              "who'll": "who shall / who will",
              "who'll've": "who shall have / who will have",
              "who's": "who has / who is",
              "who've": "who have",
              "why's": "why has / why is",
              "why've": "why have",
              "will've": "will have",
              "won't": "will not",
              "won't've": "will not have",
              "would've": "would have",
              "wouldn't": "would not",
              "wouldn't've": "would not have",
              "y'all": "you all",
              "y'all'd": "you all would",
              "y'all'd've": "you all would have",
              "y'all're": "you all are",
              "y'all've": "you all have",
              "you'd": "you had / you would",
              "you'd've": "you would have",
              "you'll": "you shall / you will",
              "you'll've": "you shall have / you will have",
              "you're": "you are",
              "you've": "you have"}

       def contract(text):
              for word in text.split():
                     if word.lower() in contractions:
                            text = text.replace(word, contractions[word.lower()])
              return text
       
       def preprocess(text_column):
              """
              Function:     This NLP pre processing function takes in a sentence,
                            replaces all the useless letters and symbols, and takes 
                            out all the stop words. This would hopefully leave only 
                            the important key words
                            
              Input:        A list of sentences
              
              Returns:      A list of sentences that has been cleaned
              """
              # Remove link,user and special characters
              # And Lemmatize the words
              new_review = []
              for review in text_column:
                     # this text is a list of tokens for the review
                     text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(review).lower()).strip()
                     text = contract(text).split(' ')
                     
                     # Stemming
                     text = [ps.stem(i) for i in text if i not in stop_words]
                     
                     new_review.append(' '.join(text))
              return new_review
       
       questions = preprocess(questions)
       if extent == 'full':
              with open('tokenizer_2020-05-05.pickle', 'rb') as handle:
                  tokenizer = pickle.load(handle)
              questions = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen = 100)
       return questions

def predict_one(question):
    question = _PREPROCESS_Q(pd.Series(question))
    pred = model.predict(question)
    return pd.DataFrame(np.vectorize(lambda x: int(x*100000)/100000)(pred), columns = col_names)
        
# code for Model 2

@app.route('/') 
@app.route('/home')
def home():                        
    return render_template('home.html') # posts acts as whatever variable that's been input into the html code

### Model 1 - Seedly Question Labelling 
@app.route('/SeedlyQuestionLabelling') 
def SeedlyQuestionLabelling():
    return render_template('SeedlyQuestionLabelling.html', 
                            title = 'Seedly Question Labelling') 

@app.route('/SeedlyQuestionLabelling', methods = ['POST']) # the posted message taken from this route
def SeedlyQuestionLabelling_post():
    text = request.form['Question'] # has to be the same as the textarea

    #result_df = _IFELSE_MODEL(_PREPROCESS_Q(text))

    result_df = predict_one(text)

    result_df = list(zip(result_df.columns, np.where(result_df.values[0] > 0.5, 1, 0)))

    return render_template('SeedlyQuestionLabelling.html', 
                            title = 'Seedly Question Labelling',
                            result = result_df) 

### Model 2 - 

# EXECUTION
if __name__ == '__main__':
    """
    doing this will allow running this script to return the url for the web app,
    instead of doing it manually on Terminal
    """
    app.run(debug = True)
    