import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
# Download NLTK resources
nltk.download('stopwords')
predictions = []
# Load models and data
file_name = "Spam_sms_prediction.pkl"
model = pickle.load(open(file_name, 'rb'))

file_name = "corpus.pkl"
corpus = pickle.load(open(file_name, 'rb'))

# Initialize CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return model.predict(temp)

def show_wordcloud(sample_message):
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = set(stopwords.words('english')), 
                min_font_size = 10).generate(sample_message)
    st.image(wordcloud.to_array())
    
# Page title and favicon
st.set_page_config(page_title="Social Sentinel", page_icon="📧")

# Custom CSS for style enhancement
st.markdown(
    """
    <style>
    .full-width {
        width: 100%;
    }
    .header {
        background-color:#f63366;
        padding:10px;
        border-radius:10px;
        margin-bottom: 20px;
    }
    .info-table {
        width: 100%;
    }
    .info-table td {
        padding: 10px;
        border: 1px solid #ddd;
        text-align: left;
    }
    .info-table th {
        padding: 10px;
        border: 1px solid #ddd;
        text-align: left;
        background-color: #f63366;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="header">
    <h1 style="color:white;text-align:center;">📬 Social Sentinel 🛡️</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Create information table
st.markdown("<h2>App Features : </h2>", unsafe_allow_html=True)
st.markdown(
    """
    <table class="info-table">
        <tr>
            <th>Feature</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Enter Message</td>
            <td>Input any text message to check if it's spam or not.</td>
        </tr>
        <tr>
            <td>Instant Feedback</td>
            <td>Receive instant feedback on whether the message is spam or not.</td>
        </tr>
        <tr>
            <td>Visual Indication</td>
            <td>See visual indicators (images) to easily identify spam or non-spam messages.</td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)
# Header for message input
st.header("Enter your message here :")

# Text area for message input
name = st.text_area("", height=160)

# Button to submit with custom styling
button_html = f'<button class="full-width" type="button" onclick="if(this.disabled)return false;this.disabled=true;form.submit()">Check for Spam</button>'
button = st.markdown(button_html, unsafe_allow_html=True)

# Predict and display result
if button:
    if name.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Prediction
        score = predict_spam(name)
        
        # Explanation for prediction
        explanation = "This message is likely spam." if score == 1 else "This message is likely not spam."
        
        # Display length of message
        st.write("Message Length:", len(name))
        
        # Display prediction and explanation
        st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)
        if score == 1:
            st.error(explanation)
            st.image('spam.png', width=300)
        else:
            st.success(explanation)
            st.image("nospam.png", width=300)
        
        # Accumulate predictions
        predictions.append(score)
        
        # Predictions distribution bar chart
        st.markdown("<h2>Predictions Distribution</h2>", unsafe_allow_html=True)
        prediction_labels = ["Non-Spam", "Spam"]
        fig, ax = plt.subplots(figsize=(8, 6))
        non_spam_count = sum(1 for x in predictions if x == 0)
        spam_count = sum(1 for x in predictions if x == 1)
        ax.bar(prediction_labels[0], non_spam_count, color="green")
        ax.bar(prediction_labels[1], spam_count, color="red")
        ax.set_title("Distribution of Predicted Spam and Non-Spam Messages")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Display word cloud
        st.markdown("<h2>Word Cloud</h2>", unsafe_allow_html=True)
        show_wordcloud(name)

# acc=pd.DataFrame({"Models":["SVC","KN","NB","DT","LR","RF"],
#                   "Accuracy":[0.962249,0.937225,0.987273,0.961292,0.953254,0.973684],
#                   "F1 Score":[0.811,0.856,0.952,0.910,0.872,0.927]})
# st.bar_chart(acc,x='Models',y=['Accuracy','F1 Score'],color=['#FFC000','#6b04fd'])