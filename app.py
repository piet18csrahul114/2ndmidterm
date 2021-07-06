import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle
app = Flask(__name__)
from keras.models import load_model
model = load_model('modelANN.h5')
# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, :-1].values
# Extracting dependent variable:
y = dataset.iloc[:, 8].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 2:7]) 
#Replacing missing data with the calculated mean value  
X[:, 2:7]= imputer.transform(X[:, 2:7]) 

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Splitting the Dataset into the Training set and Test set

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Feature Scaling
# Standard Scaling:  Standardization = X'=X-mean(X)/standard deviation
# normal scaling : Normalization= X'=X-min(X)/max(x)-min(X)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  Glucose = int(request.args.get('Glucose'))
  Age = int(request.args.get('Age'))
  Gender = int(request.args.get('Gender'))
  BMI = float(request.args.get('BMI'))  
  Insulin = int(request.args.get('Insulin'))
  DiabetesPedigreeFunction = float(request.args.get('DiabetesPedigreeFunction')) 
  SkinThickness = int(request.args.get('SkinThickness'))
  BloodPressure = int(request.args.get('BloodPressure'))
   
  
  output=model.predict(sc_X.transform([[Gender,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]))
  output = (output > 0.5)
  if output>0.5:
    result='He will be diabetic'
  else:
    result='He will not be diabetic'
        
  return render_template('index.html', prediction_text='Model  has predicted  : {}'.format(result))


app.run()