import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model, scaler, and label encoders
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_conversion(input_data):
    # List of numeric columns
    numeric_columns = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
                       'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
                       'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints']
    # Scaling numerical features
    input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])
    
    # Make prediction
    prediction = model.predict(input_data)
    return 'Can Be Convert' if prediction[0] == 1 else 'will Not be Convert'

# Streamlit App
st.title("Marketing Campaign Conversion Prediction")

# User input form
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Male', 'Female'])
income = st.number_input('Income', min_value=10000, max_value=200000)
campaign_channel = st.selectbox('Campaign Channel', ['Social Media', 'Email', 'PPC', 'SEO', 'Referral'])
campaign_type = st.selectbox('Campaign Type', ['Awareness', 'Retention', 'Conversion', 'Consideration'])
ad_spend = st.number_input('Ad Spend', min_value=0.0, max_value=10000.0)
click_through_rate = st.number_input('Click Through Rate', min_value=0.0, max_value=1.0)
conversion_rate = st.number_input('Conversion Rate', min_value=0.0, max_value=1.0)
website_visits = st.number_input('Website Visits', min_value=0)
pages_per_visit = st.number_input('Pages Per Visit', min_value=0.0, max_value=20.0)
time_on_site = st.number_input('Time on Site (minutes)', min_value=0.0, max_value=60.0)
social_shares = st.number_input('Social Shares', min_value=0)
email_opens = st.number_input('Email Opens', min_value=0)
email_clicks = st.number_input('Email Clicks', min_value=0)
previous_purchases = st.number_input('Previous Purchases', min_value=0)
loyalty_points = st.number_input('Loyalty Points', min_value=0)

# Mapping categorical variables
gender_mapping = {'Male': 0, 'Female': 1}
channel_mapping = {'Email': 0, 'PPC': 1, 'Referral': 2, 'SEO': 3, 'Social Media': 4}
campaign_type_mapping = {'Awareness': 0, 'Retention': 3, 'Consideration': 1, 'Conversion': 2}

gender = gender_mapping[gender]
campaign_channel = channel_mapping[campaign_channel]
campaign_type = campaign_type_mapping[campaign_type]

# Collecting input data as a single-row DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Income": [income],
    "CampaignChannel": [campaign_channel],
    "CampaignType": [campaign_type],
    "AdSpend": [ad_spend],
    "ClickThroughRate": [click_through_rate],
    "ConversionRate": [conversion_rate],
    "WebsiteVisits": [website_visits],
    "PagesPerVisit": [pages_per_visit],
    "TimeOnSite": [time_on_site],
    "SocialShares": [social_shares],
    "EmailOpens": [email_opens],
    "EmailClicks": [email_clicks],
    "PreviousPurchases": [previous_purchases],
    "LoyaltyPoints": [loyalty_points]
})

# Predict button
if st.button('Predict Conversion'):
    result = predict_conversion(input_data)
    st.write(f'The prediction is: {result}')
