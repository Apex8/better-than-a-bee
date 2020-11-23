import flask
from flask import render_template
import joblib
import sklearn
import numpy as np
import pandas as pd
import os
import glob
from joblib import load
from fastai.vision.all import *
from werkzeug.utils import secure_filename
from datetime import datetime

app = flask.Flask(__name__, template_folder='templates')

# Config settings
app.config["IMAGE_UPLOADS"] = "static/img"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "JFIF"]
app.config["MAX_IMAGE_FILESIZE"] = 250000

# Function to check file extension (imgrecognition)
def allowed_image(filename):

    if not "." in filename:
        return False

    global ext
    ext = filename.rsplit(".", 1)[1]
    
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

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

@app.route('/viz5')
def viz5():
    return render_template('viz5.html')

@app.route('/viz6')
def viz6():
    return render_template('viz6.html')

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

@app.route('/imgrecognition', methods=['GET', 'POST'])

def img_predict():

    # Run on Submit button click
    if flask.request.method == 'POST':

        # This works, but is clumsy - can't figure out a better way to do it
        # Without this the file caches and doesn't update on reload
        # Identify the file to be deleted
        del_file = glob.glob(app.config["IMAGE_UPLOADS"] + '/UPLOAD_PIC*')
        # Convert to a string and remove the root
        file_to_delete = str(del_file)[14:-2]

        try:
            # Delete the previously uploaded file
            os.remove(app.config["IMAGE_UPLOADS"] + '/' + file_to_delete)
        
        except:
            print('')

        # Get the file name
        image = flask.request.files['image']

        # Get the image file size
        image.seek(0, os.SEEK_END)
        size = image.tell()
        
        # If file size > size limit in config settings then do not accept the image
        if size > app.config["MAX_IMAGE_FILESIZE"]:
            return(flask.render_template('imgrecognition.html', prediction="Maximum file size exceeded."))
        
        if allowed_image(image.filename):

            now = datetime.now()
            substr_now = str(now)[-6:]

            # Use a constant filename, but with a variable extension - this facilitates deletion later
            filename = 'UPLOAD_PIC' + substr_now + '.' + ext
            
            # fastai creates an image object - not really sure why this is necessary
            img = PILImage.create(image)

            # Save the image
            img.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
    
            # Load the model
            learn = load_learner('model/export.pkl')
                    
            # Run the image through the model
            pred_class, pred_idx, outputs = learn.predict(img)

            # Return the prediction on the webpage and display the image
            return(flask.render_template('imgrecognition.html', prediction=f'Prediction class: {pred_class}', selected_image=app.config["IMAGE_UPLOADS"] + '/' + filename))
            
        else:
            return(flask.render_template('imgrecognition.html', prediction="Please select a valid file type."))
    
    if flask.request.method == 'GET':
        
        return(flask.render_template('imgrecognition.html'))

if __name__ == '__main__':
    app.run()
