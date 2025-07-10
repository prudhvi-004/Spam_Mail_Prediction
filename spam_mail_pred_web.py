import numpy as np
import pickle
import streamlit as st

# ✅ Load trained model
with open('spam_mail_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# ✅ Load trained vectorizer
with open('vectorizer.pkl', 'rb') as f:
    feature_extraction = pickle.load(f)

def spam_mail_prediction(input_data):
    # ✅ Transform input using trained vectorizer
    input_data_features = feature_extraction.transform(input_data)

    # Predict using trained model
    prediction = loaded_model.predict(input_data_features)

    if prediction[0] == 1:
        return "Ham mail"
    else:
        return "Spam mail"

def main():
    st.title('Spam Mail Prediction')
    
    Message = st.text_input('Enter the Mail')
    
    prediction = ''
    
    if st.button('Spam Or Ham'):
        prediction = spam_mail_prediction([Message])
        
    st.success(prediction)

if __name__ == '__main__':
    main()
