import streamlit as st
import pandas as pd
import pickle
df = pd.read_csv(r'cleaned_data.csv')
st.title('Bangalore House Prediction')
location = st.selectbox('Location', df['location'].unique())
sqft = st.slider('square feet', min_value = 100, max_value = 30000, value = 1500)
bath = st.slider('Bathroom', min_value = 0, max_value = 20, value = 2)
bhk = st.slider('BHK', min_value = 1, max_value = 20, value = 3)

df2 = pd.DataFrame({
    'location' : [location],
    'total_sqft' : [sqft],
    'bath' : [bath],
    'bhk' : [bhk]
})
with open('Bangalore House Prediction.pkl', 'rb') as f:
    model = pickle.load(f)
    
if st.button('Predict Price'):
    output = model.predict(df2)
    st.success(f'Predicted Price is ₹{1e5*(output[0]):.2f}')
    