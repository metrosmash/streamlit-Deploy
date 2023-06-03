import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

import pickle as pc

st.title('NY2017 Batch Prediction')
#loading the model
model_in = open('metrosmash/streamlit-Deploy/models/logreg.pkl','rb')
model = pc.load(model_in)

#st.title('NY2017 Batch Prediction')

dataf = st.file_uploader('Upload Your CSV File Here')

if dataf is not None:
    def load_data(file):
        df = pd.read_csv(file,low_memory=False)
        
        return df 

data = load_data(dataf)
#st.write(f'load data-shape:{data.shape}')
def dropp(file):
    file =  file.drop(['Birth Weight','Payment Typology 3','Zip Code - 3 digits','Operating Certificate Number','Facility Name','Permanent Facility Id','Discharge Year','CCS Diagnosis Description','CCS Procedure Description', 'APR DRG Description','APR MDC Description'], axis = 1)
    return file

#data = dropp(data)
#st.write(f'drop data-shape:{data.shape}')
cat_list = ['Hospital Service Area', 'Hospital County', 'Age Group', 'Gender', 'Race', 'Ethnicity', 'Type of Admission', 'Patient Disposition',  'APR Severity of Illness Description', 'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1', 'Payment Typology 2', 'Abortion Edit Indicator', 'Emergency Department Indicator']
num_list = [ 'CCS Diagnosis Code', 'CCS Procedure Code', 'APR DRG Code', 'APR MDC Code', 'APR Severity of Illness Code']

def Clean_nan(file):
    num_imp = SimpleImputer(strategy = 'median')
    cat_imp = SimpleImputer(strategy = 'most_frequent')
    file[cat_list] = cat_imp.fit_transform(file[cat_list])
    file[num_list] = num_imp.fit_transform(file[num_list]) 
    return file

#data = Clean_nan(data)
#st.write(f'clean-nan-shape:{data.shape}')

cat_list2 = ['Hospital Service Area', 'Hospital County', 'Age Group', 'Gender', 'Race', 'Ethnicity', 'Type of Admission', 'Patient Disposition', 'APR Severity of Illness Description', 'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1', 'Payment Typology 2', 'Abortion Edit Indicator', 'Emergency Department Indicator']


def dummies_f(file):
    file = pd.get_dummies(data = file,drop_first = True,sparse = True)
    return file 

#data = dummies_f(data)
#st.write(f'dummies-shape:{data.shape}')


pred = model.predict(data)
data['predict lengthofstay'] = pred
st.write('PREVIEW',data.head()) #its bringing out error because its not large enough

#The Download function is implemented below
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(data)
st.write('Download the predicted file here')
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='predicted.csv',
    mime='text/csv',
)
