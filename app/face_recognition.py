import numpy as np
import sklearn
import pickle
import cv2
import os

#print(os.getcwd())

# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.txt')
model_svm = pickle.load(open('./model/model_svm.pickle', mode='rb'))
pca_model = pickle.load(open('./model/pca_dict.pickle', mode = 'rb')) 

model_pca = pca_model["pca"] 
mean_face = pca_model["mean_face"]


def face_recognition_pipeline(filename,path=True):
    if path:  # 01: Read in an image
        # step-01: read image
        img = cv2.imread(filename) # BGR
    else:
        img = filename # array
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 02: Convert to grayscale 
    faces = haar.detectMultiScale(img_gray,1.5,3)  # 03: Crop the face using Haar Cascade Classifier 

    predictions = []
    for x,y,w,h in faces:
        roi = img_gray[y:y+h,x:x+w]  # 03: Crop the face using Haar Cascade Classifier
        roi = roi/255.0   # 04: normalise (0-1) 

        if roi.shape[0] < 144:
            roi= cv2.resize(roi,(144,144),cv2.INTER_AREA)  # 05: Resize to 144x144 
        else:
            roi = cv2.resize(roi,(144,144),cv2.INTER_CUBIC)

        # 06: Flatten 
        roi = roi.reshape(1,-1) # flattens but makes 2-d, unlike flatten() which is 1-d, 2-d needed for PCA
        roi -= mean_face  # 07: Subtract mean face 

        eigen_face = model_pca.transform(roi)  # 08: Exctract feutures, get eigen image 
        eig_img = model_pca.inverse_transform(eigen_face)  # step-09 Eigen Image for Visualization
        results = model_svm.predict(eigen_face)
        prob_score = model_svm.predict_proba(eigen_face)
        prob_score_max = prob_score.max()
        text = f'{results[0]}: {prob_score_max*100:.1f}%'

        if results[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,5)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)  # fills
        cv2.putText(img,text,(x+10,y),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),5)

        output = {
            'roi':roi,
            'eig_img':eig_img,
            'prediction_name':results[0],
            'score':prob_score_max
        }
        predictions.append(output)
    return img,predictions

