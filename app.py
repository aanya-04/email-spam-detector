# import nltk
# import pickle
# import streamlit as st
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import string
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split

# nltk.download('punkt')
# nltk.download('stopwords')
# ps = PorterStemmer()

# # Function remains unchanged
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = [ps.stem(word) for word in y if word not in stopwords.words('english') and word not in string.punctuation]
#     return " ".join(text)

# # New: Train and save model if needed
# data = pd.read_csv("spam.csv", encoding="latin-1")
# data = data[['v1', 'v2']]
# data.columns = ['label', 'text']
# data['label'] = data['label'].map({'spam': 1, 'ham': 0})
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
# X_train_vectorized = vectorizer.fit_transform(X_train)

# model = MultinomialNB()
# model.fit(X_train_vectorized, y_train)

# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# # Load the trained model and vectorizer
# tk = pickle.load(open("vectorizer.pkl", 'rb'))
# model = pickle.load(open("model.pkl", 'rb'))

# st.title("Email Spam Detection")

# input_sms = st.text_input("Enter the Email")

# if st.button('Predict'):
#     transformed_sms = transform_text(input_sms)
#     vector_input = tk.transform([transformed_sms])
#     result = model.predict(vector_input)[0]
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")


import nltk
import pickle
import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.data import find

# Ensure NLTK resources are available
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')


try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = [ps.stem(word) for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(text)

# Load and prepare data
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'spam': 1, 'ham': 0})
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Train and save model
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Load model and vectorizer
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit App
st.title("📧 Email Spam Detector")

input_sms = st.text_input("Enter the Email")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tk.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("⚠️ Spam")
    else:
        st.header("✅ Not Spam")

