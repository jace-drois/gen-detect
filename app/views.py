from flask import render_template, request
import os
import cv2
from app.face_recognition import face_recognition_pipeline as frp
from datetime import datetime
import matplotlib.image as matim

upload_folder = 'static/upload'

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method == "POST":
        f = request.files['image_name']
        filename = f.filename
        path = os.path.join(upload_folder,filename)
        f.save(path) # saves image to upload folder

        pred_image, predictions = frp(filename=path)
        #now = datetime.now().strftime("%H%M%S")
        #pred_filename = f'prediction_image_{now}.jpg'
        pred_filename = f'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)

        # the returned predictions contains a dictionary in a list
        # for every detected face in image with following keys
        # 'roi' - the grayscale image, vector of 144x144 values
        # 'eig_img' - the eigen face, vector of 144x144 values
        # 'prediction_name' - 'male' or 'female'
        # 'score'  - percentage

        # generate report
        report = []
        for i, obj in enumerate(predictions):
            gray_img = obj['roi'].reshape(144,-1)
            eigen_img = obj['eig_img'].reshape(144,144)
            gender_name = obj['prediction_name']
            score = round(obj['score']*100,2)

            gray_name = f'roi_{i}.jpg'
            eig_name = f'eigen_{i}.jpg'
            # OpenCv only saves image 8 bit array with values between 0 and 255
            # in order to save with values betwenn 0 and 1, use Matplotlib
            matim.imsave(f'./static/predict/{gray_name}',gray_img,cmap='gray')
            matim.imsave(f'./static/predict/{eig_name}',eigen_img,cmap='gray')

            report.append([gray_name, eig_name, gender_name, score])
        return render_template('gender.html',fileupload=True,report=report) # POST REQUEST

    return render_template('gender.html',fileupload=False)  # GET REQUEST

