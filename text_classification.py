import streamlit as st
import requests

# API details
API_URL = "https://api-inference.huggingface.co/models/KiDoAIO/my_awesome_model"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit app
st.title("Demo Sentiment Analysis for Delta Brains JSC.")

st.write("Enter some text to send to the model:")

user_input = st.text_area("Input Text", "I like you. I love you")

if st.button("Submit"):
    output = query({"inputs": user_input})

    if isinstance(output, dict) and "error" in output:
        st.error(output["error"])
    elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], list) and len(output[0]) > 0:
        # Assuming the response is a nested list of dictionaries
        prediction = output[0][0]  # Take the first prediction in the nested list
        label = prediction.get('label', 'No label')
        score = prediction.get('score', 'No score')
        
        st.write(f"Result: {label}")
        st.write(f"Accuracy: {score:.2f}")
    else:
        st.error("Unexpected response format")

# To run the app, save this script and use the command: streamlit run app.py
