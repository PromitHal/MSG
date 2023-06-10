import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from Data_loader import Data
from Preprocess_Data import Data_Modify
from sklearn.preprocessing import StandardScaler
from Evaluation import Evaluate
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
path_filename=r'C:\Users\PROMIT\Desktop\Music_Genre\music_01.csv'
label_dict={
    'blues':0,
    'classical':1,
    'country':2,
    'disco':3,
    'hiphop':4,
    'jazz':5,
    'metal':6,
    'pop':7,
    'reggae':8,
    'rock':9
}
num_epochs=40
data=Data(path_filename,label_dict=label_dict)

trainx,trainy,testx,testy,valx,valy=data.split()

sc=StandardScaler()

trainx=sc.fit_transform(trainx)
testx=sc.transform(testx)


knn=KNeighborsClassifier()
knn.fit(trainx,trainy)

preds=knn.predict(testx)

ev=Evaluate(preds,testy)
ev.calc_eval()