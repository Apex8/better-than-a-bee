import flask
from flask import render_template
import joblib
import sklearn
import numpy as np
import pandas as pd
from joblib import load
# X = pesticide amount, honeybee colony number, y = total honey production
 

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/imgrecognition')
def imgrecognition():
    return render_template('imgrecognition.html')

@app.route('/references')
def references():
    return render_template('references.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/viz1')
def viz1():
    return render_template('viz1.html')

@app.route('/viz2')
def viz2():
    return render_template('viz2.html')

@app.route('/viz3')
def viz3():
    return render_template('viz3.html')

@app.route('/viz4')
def viz4():
    return render_template('viz4.html')

@app.route('/model', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'POST':
        LRE_model = joblib.load('model/kaggle_LRE_Model.sav')
        kaggle_model = joblib.load('model/kaggle_model.sav')
        # brain_model is saved from 
        X_scaler = joblib.load('model/X_scaler.sav')
        # y_scaler = joblib.load('y_scaler.sav')
        nCLOTHIANIDIN = flask.request.form['nCLOTHIANIDIN']
        nIMIDACLOPRID = flask.request.form['nIMIDACLOPRID']
        nTHIAMETHOXAM = flask.request.form['nTHIAMETHOXAM']
        nACETAMIPRID = flask.request.form['nACETAMIPRID']
        nTHIACLOPRID = flask.request.form['nTHIACLOPRID']
        nAllNeonic = flask.request.form['nAllNeonic']
        numcol = flask.request.form['numcol']
        # img = flask.request.form['img']
        X = pd.DataFrame({'nCLOTHIANIDIN': [nCLOTHIANIDIN], 
                        'nIMIDACLOPRID': [nIMIDACLOPRID],
                        'nTHIAMETHOXAM': [nTHIAMETHOXAM],
                        'nACETAMIPRID': [nACETAMIPRID],
                        'nTHIACLOPRID': [nTHIACLOPRID],
                        'nAllNeonic': [nAllNeonic],
                        'numcol': [numcol]})
        print(X)
        X_scaled = X_scaler.transform(X)
        print(X_scaled)

        scaled_y = LRE_model.predict(X_scaled)[0][0]
        y = LRE_model.predict(X)[0][0]
        kaggle_y = kaggle_model.predict(X)[0][0]

        return(flask.render_template('model.html', prediction=y, scaled_prediction=scaled_y, kaggle_y=kaggle_y))
    if flask.request.method == 'GET':
        return(flask.render_template('model.html'))
if __name__ == '__main__':
    app.run()
