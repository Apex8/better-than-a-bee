import flask
import joblib
import sklearn
import numpy as np
import pandas as pd
# X = pesticide amount, honeybee colony number, y = total honey production
 

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'POST':
        model = joblib.load('brain_model.sav')
        # brain_model is saved from 
        #scalar = joblib.load('X_scalar.sav')
        weight = flask.request.form['weight']
        # img = flask.request.form['img']
        X = pd.DataFrame({'weight': [weight]})
        #scalar.transform(X)
        y = model.predict(X)[0][0]
        return(flask.render_template('main.html', prediction=y))
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run()