# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:29:29 2022

@author: HP
"""

from flask import Flask,jsonify,request,render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''For rendering results on html gui'''
    int_features = [int (x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html',prediction_text = 'Pump status {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)