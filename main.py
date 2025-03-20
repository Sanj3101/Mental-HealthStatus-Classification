import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Dropdown for selecting model
model_choice = st.selectbox("Choose Model", ["BERT", "DistilBERT"])

# Load model and tokenizer dynamically
model_name = "saved_mental_status_bert" if model_choice == "BERT" else "saved_mental_status_distbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load label encoder and define label mapping
label_mapping = {
    0: 'Anxiety',
    1: 'Bipolar',
    2: 'Depression',
    3: 'Normal',
    4: 'Personality Disorder',
    5: 'Stress',
    6: 'Suicidal'
}

# Custom function
stop_words = set(stopwords.words('english'))

def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    cleaned_statement = ' '.join(words)
    return cleaned_statement

# Detection System (Example)
def detect(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map predicted class to corresponding label
    return label_mapping.get(predicted_class, "Unknown Status")

# UI app
st.title("Mental Health Status Detection")

input_text = st.text_input("Enter Your mental state here....")

if st.button("Detect"):
    predicted_class = detect(input_text)
    st.write("Predicted Status:", predicted_class)
