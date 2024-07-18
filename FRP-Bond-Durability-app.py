import pandas as pd
import streamlit as st
from PIL import Image
import pickle


st.write(
    "<h1 style='color:blue;font-size:30px;text-align:center'>Prediction of FRP-to-Concrete Bond Strength Under Different Exposure Conditions</h1>",
    unsafe_allow_html=True,
)

st.write('---')

image = Image.open('Flowchart-2.png')
width, height = image.size
new_size = (int(width * 0.8), int(height * 0.8))  # Reduce the size by 20%
image.thumbnail(new_size)
st.image(image, use_column_width=True) 

# Loads the FRP-RC columns Dataset
csc = pd.read_excel("FRP-Bond-Durability.xlsx", usecols="A:K", header=0)

# Convert data
csc['fc'] = csc['fc'].astype(float)
csc['bc'] = csc['bc'].astype(float)
csc['Lf'] = csc['Lf'].astype(float)
csc['bf'] = csc['bf'].astype(float)
csc['tf'] = csc['tf'].astype(float)
csc['Ef'] = csc['Ef'].astype(float)
csc['ff'] = csc['ff'].astype(float)
csc['EtEC'] = csc['EtEC'].astype(float)
csc['TRH'] = csc['TRH'].astype(float) 
csc['D'] = csc['D'].astype(float)
csc['Pu'] = csc['Pu'].astype(float)

csc = csc[['fc', 'bc', 'Lf', 'bf', 'tf', 'Ef', 'ff', 'EtEC', 'TRH', 'D', 'Pu']]
y = csc['Pu'].copy()
X = csc.drop('Pu', axis=1).copy()


# Header of Input Parameters
st.sidebar.header('Input Parameters')
value = ("0", "1")
options = list(range(len(value)))



def input_variable():
    fc = st.sidebar.slider('CS of Concrete (MPa)', float(csc['fc'].min()), float(csc['fc'].max()),
                           float(csc['fc'].mean()))
    bc = st.sidebar.slider('Width of concrete block (mm)', float(csc['bc'].min()), float(csc['bc'].max()),
                            float(csc['bc'].mean()))
    Lf = st.sidebar.slider('Bonded length (mm)', float(csc['Lf'].min()), float(csc['Lf'].max()),
                            float(csc['Lf'].mean()))
    bf = st.sidebar.slider('Bonded width (mm)', float(csc['bf'].min()), float(csc['bf'].max()),
                            float(csc['bf'].mean()))
    tf = st.sidebar.slider('Thickness of FRP composite (mm)', float(csc['tf'].min()), float(csc['tf'].max()),
                            float(csc['tf'].mean()))
    Ef = st.sidebar.slider('Elastic Modulus of FRP composite (GPa)', float(csc['Ef'].min()), float(csc['Ef'].max()),
                            float(csc['Ef'].mean()))
    ff = st.sidebar.slider('Tensile strength of FRP composite (MPa)', float(csc['ff'].min()), float(csc['ff'].max()),
                            float(csc['ff'].mean()))
    EtEC = st.sidebar.slider('Ratio of exposure type to exposure condition', float(csc['EtEC'].min()), float(csc['EtEC'].max()),
                            float(csc['EtEC'].mean()))
    TRH = st.sidebar.slider('Ratio of temperature to relative humidity (oC/%)', float(csc['TRH'].min()), float(csc['TRH'].max()),
                            float(csc['TRH'].mean()))
    D = st.sidebar.slider('Exposure duration', float(csc['D'].min()), float(csc['D'].max()),
                            float(csc['D'].mean()))
   

    data = {
            'fc':  fc,
            'bc' : bc,
            'Lf' : Lf,
            'bf' : bf,
            'tf' : tf,
            'Ef' : Ef,
            'ff' : ff,
            'EtEC' : EtEC,
            'TRH' : TRH,
            'D' : D,
            }

    features = pd.DataFrame(data, index=[0])
    return features

df = input_variable()

st.header('Specified Input Parameters')
st.write(df)
st.write('---')

xgb_model = pickle.load(open('FRP-Bond-Durability_xgb_model.pkl', 'rb'))

prediction = xgb_model.predict(df)[0]



st.markdown(
    "<h2 style='text-align:justify'>Predicted FRP-to-Concrete Bond Strength Under Different Exposure Conditions</h2>",
    unsafe_allow_html=True
)
st.write('<font color="green"><b>$P_{u,pred}$ = ' + str(prediction) + ' kN</b></font>', unsafe_allow_html=True) 

st.write('---')


st.markdown('## **SHAP Summary Plot**')
image = Image.open('SHAP_relative_importance-1.png')
st.image(image, use_column_width=True)
st.write('---')

st.markdown('## **Taylor Plot**')
image = Image.open('Taylorplot.png')
st.image(image, use_column_width=True)
st.write('---')


st.write("""
This GUI predicts the FRP-to-Concrete Bond Strength Under Different Exposure Conditions (Developed by **Aman Kumar**, under the guidance of **Dr. Harish Chandra Arora**, <span style='color:blue; font-style: italic'>Structural Engineering Department</span>)
***
""", unsafe_allow_html=True)
