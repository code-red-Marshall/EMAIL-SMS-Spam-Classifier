import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure the necessary NLTK resources are downloaded
nltk_packages = ['punkt', 'stopwords']
for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model from the pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for the message
input_sms = st.text_area("Enter the message")

# Button to check if the message is spam or not
if st.button('Check'):

    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict using the pre-trained model
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("It's a Spam. Beware!")
    else:
        st.header("Not Spam. It's safe")
