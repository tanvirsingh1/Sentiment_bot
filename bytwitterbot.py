import nltk
import streamlit as st
import os
import pickle
import joblib
from dotenv import load_dotenv
import google.generativeai as genai
import praw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

st.set_page_config(page_title="Sentiment Analysis, Q&A Bot and Hashtag Recommendation")

# Load the environment variables
load_dotenv()
# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini chat
gemini_model = genai.GenerativeModel("gemini-pro")
chat = gemini_model.start_chat(history=[])

reddit = praw.Reddit(
    client_id=os.getenv(("Client_ID")),
    client_secret=os.getenv(("Client_Secret")),
    user_agent=os.getenv(("Due_Plastic_4892")),
)


nltk.download('punkt')
nltk.download('averaged_perceptro n_tagger')
#"Looking for the perfect pair of shoes? Check out our latest collection! Stylish, comfortable, and affordable. Don't miss out – limited stock available!"
# Why should I check the sentiment of my tweet, and why are hashtags important? Explain very briefly.
#Tweet: I can't believe how awful this customer service is. I've been waiting for hours, and still no response!
# Function to extract keywords from text using NLTK


def extract_keywords(content):
    tokens = nltk.word_tokenize(content)
    tags = nltk.pos_tag(tokens)
    keywords = [
        word for word, tag in tags
        if (tag.startswith('NN') or tag == 'JJ') and re.match(r'\w+', word)
    ]
    return keywords

# Function to search Reddit for trending keywords
def search_reddit_for_keywords(keywords, max_results=10):
    trending_keywords = []

    for keyword in keywords:
        for submission in reddit.subreddit('all').search(keyword, limit=10, time_filter='day'):
            post_content = submission.title + " " + submission.selftext
            trending_keywords += extract_keywords(post_content)

    keyword_counts = Counter(trending_keywords)
    top_keywords = keyword_counts.most_common(max_results)
    return [keyword for keyword, _ in top_keywords]

def get_gemini_hashtags(keywords):
    prompt = f"Generate 10 trending hashtags for the following keywords: {', '.join(keywords)}"
    response = chat.send_message(prompt, stream=False)

    response_text = response.text.strip()

    # Extracting hashtags directly from the response text
    # We assume the response contains hashtags like "#hashtag1 #hashtag2"
    hashtags = re.findall(r'#\w+', response_text)

    return hashtags


# Function to filter relevant keywords using semantic similarity
def filter_relevant_keywords(input_text, reddit_keywords):
    corpus = [input_text] + reddit_keywords
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectorizer)
    scores = similarity_matrix[0, 1:]
    keyword_scores = zip(reddit_keywords, scores)
    return [keyword for keyword, score in keyword_scores if score > 0.2]

def generate_combined_hashtags(content):
    keywords = extract_keywords(content)
    reddit_keywords = search_reddit_for_keywords(keywords, max_results=10)
    relevant_reddit_keywords = filter_relevant_keywords(content, reddit_keywords)
    gemini_hashtags = get_gemini_hashtags(keywords)
    combined = set(gemini_hashtags + [f"#{kw}" for kw in relevant_reddit_keywords])
    return sorted(combined)

# Load the trained sentiment analysis model and vectorizer
@st.cache_resource
def load_model():
    modelfilename = 'trained_model.sav'
    vectorizerfilename = 'tfidf_vectorizer.pkl'

    # Load the sentiment analysis model
    with open(modelfilename, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")

    # Load the vectorizer
    with open(vectorizerfilename, 'rb') as vectorizer_file:
        vectorizer = joblib.load(vectorizer_file)
    print("Vectorizer loaded successfully.")

    return model, vectorizer


model, vectorizer = load_model()


# Function to get Gemini response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response


# Streamlit UI

st.title("Sentiment Analysis and AI Chatbot")
st.write("Analyze the sentiment of tweets and ask questions to the Gemini AI.")

# Selectbox for navigating between pages (Chat or Chat History)
page = st.selectbox("Choose a page", ["Chat", "Chat History"])

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Chat Page
if page == "Chat":
    show_instructions = st.checkbox("Show Instructions", value=True)

    if show_instructions:
       # st.write("### Twitter Sentiment Analysis Bot")
        st.info(
            """
            Welcome! Here's how you can interact with the bot:

            1. **Check the Sentiment of Tweets**: 
               - Type a tweet starting with `Tweet:`, and I'll tell you if it's positive or negative.

            2. **Get Trending Hashtags for Your Tweet**: 
               - After analyzing the tweet, you can click on **Generate Hashtags** to get trending hashtags based on your tweet.

            3. **Ask Anything to Our Bot**: 
               - You can ask the bot any question, and it will respond with insightful answers!

            4. **View Your Chat History**: 
               - Go to the **Chat History** tab to see all your previous conversations with the bot.
            """
        )
    # User input for tweet sentiment analysis
    user_input = st.text_input("Enter tweet text or ask a question:")

    if user_input:
        # Check if the input is a tweet for sentiment analysis
        if user_input.startswith("Tweet:"):
            # Perform sentiment analysis on the tweet
            tweet_message = user_input[6:].strip()  # Remove "Tweet:" prefix
            tweet_vectorized = vectorizer.transform([tweet_message])
            sentiment = model.predict(tweet_vectorized)[0]

            # Map sentiment to positive/negative
            sentiment_label = "positive" if sentiment == 1 else "negative"

            # Respond with sentiment analysis
            sentiment_response = f"This tweet is {sentiment_label}."
            st.subheader("Response :")
            st.write(f"**You:** {tweet_message}")
            st.write(f"**Bot:** {sentiment_response}")
            st.session_state['chat_history'].append(("You", user_input))
            st.session_state['chat_history'].append(("Bot", sentiment_response))

            if st.button("Generate Hashtags"):
                hashtags = generate_combined_hashtags(tweet_message)
                st.subheader("Generated Hashtags:")
                st.write(", ".join(hashtags))
                st.session_state['chat_history'].append(("Bot Recommended Hashtags: ", hashtags))
                # Combine tweet message with hashtags
                tweet_with_hashtags = f"{tweet_message} {' '.join([hashtag for hashtag in hashtags])}"

                # Display tweet with hashtags
                st.subheader("Recommended Tweet with Hashtags:")
                st.write(tweet_with_hashtags)

                # Add the recommended tweet with hashtags to chat history
                st.session_state['chat_history'].append(("Bot", tweet_with_hashtags))


        else:
            # Process general questions with Gemini AI
            response = get_gemini_response(user_input)

            # Add user input and response to chat history
            st.session_state['chat_history'].append(("You", user_input))

            st.subheader("Response :")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))



# Chat History Page
elif page == "Chat History":
    st.subheader("Chat History: ")

    if len(st.session_state['chat_history']) == 0:
        st.write("No chat history yet.")
    else:
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")

    # Button to clear history
    if st.button("🗑️ Clear Chat History"):
        del st.session_state['chat_history']
        st.write("Chat history has been cleared.")
        st.rerun()  # Reload to update the UI after clearing history
