from flask import Flask, request, jsonify
import json
import random
import string
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
nltk.download('popular', quiet=True)

app = Flask(__name__)

# Load city data
try:
    with open('egypt_cities.json', 'r', encoding='utf-8') as f:
        cities_data = json.load(f)
except FileNotFoundError:
    print("Error: 'egypt_cities.json' not found.")
    cities_data = []

# Load chatbot corpus
try:
    with open('chatbot.txt', 'r', encoding='utf8') as f:
        raw_corpus = f.read().lower()
except FileNotFoundError:
    print("Warning: 'chatbot.txt' not found.")
    raw_corpus = ""

sent_tokens = nltk.sent_tokenize(raw_corpus)

# Initialize NLP tools
lemmer = WordNetLemmatizer()
remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

def LemNormalize(text):
    return [lemmer.lemmatize(word) for word in nltk.word_tokenize(text.lower().translate(remove_punct_dict))]

# Greeting detection
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey" , "Hello" , "Hey" , "Hi")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(text):
    return random.choice(GREETING_RESPONSES) if any(word in text.split() for word in GREETING_INPUTS) else None

# General chatbot response
def generate_response(user_message):
    if not raw_corpus or not user_message.strip():
        return "I don't have enough information to respond to that."

    sent_tokens.append(user_message)
    tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english').fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    score = vals[0][idx]

    sent_tokens.pop()

    return sent_tokens[idx] if score > 0 else "I’m not sure I understand. Could you rephrase that?"

# Handle city or attraction queries
def city_info_response(message):
    message = message.lower()
    for city_item in cities_data:
        if not isinstance(city_item, dict) or 'city' not in city_item:
            continue

        city = city_item['city'].lower()

        if city in message:
            if 'where' in message or 'location' in message:
                return f"{city_item['city']} is located at: {city_item['location']}."
            elif any(kw in message for kw in ['what', 'description', 'info', 'tell me about']):
                return f"{city_item['city']} is: {city_item['description']}"
            elif any(kw in message for kw in ['visit', 'attractions', 'places', 'see' ,'places to visit' , 'iconic' , 'top places' ]):
                return f"Top places in {city_item['city']}: {', '.join(city_item.get('attractions', []))}"
            else:
                return f"{city_item['city']}: {city_item['description']} Located at: {city_item['location']}."

        # Check attractions
        for attraction in city_item.get('attractions', []):
            if attraction.lower() in message:
                return f"{attraction} is located in {city_item['city']} — {city_item['description']}"

    return None

# Flask routes
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a valid message."})

    if user_message.lower() in ["bye"]:
        return jsonify({"response": "Bye! Take care."})
    if user_message.lower() in ["thanks", "thank you"]:
        return jsonify({"response": "You’re welcome!"})

    for handler in [greeting, city_info_response, generate_response]:
        response = handler(user_message)
        if response:
            return jsonify({"response": response})

    return jsonify({"response": "I'm not sure how to respond to that."})

# Evaluation tool
# def evaluate_chatbot():
    # test_data = [
    #     {"input": "Where is Cairo?", "expected_keywords": ["cairo", "located"]},
    #     {"input": "Tell me about Luxor", "expected_keywords": ["luxor", "description"]},
    #     {"input": "hi", "expected_keywords": ["hi", "hello", "hey"]},
    #     {"input": "What are some attractions in Aswan?", "expected_keywords": ["aswan", "attractions"]},
    #     {"input": "What is the weather in Mars?", "expected_keywords": ["not sure", "don't understand", "sorry"]}
    # ]

    # print("\n--- Chatbot Evaluation ---")
    # passed = 0
    # for test in test_data:
    #     message = test["input"]
    #     expected = test["expected_keywords"]
    #     response = (greeting(message) or city_info_response(message) or generate_response(message)).lower()
    #     if any(kw in response for kw in expected):
    #         print(f"[✓] {message} → {response}")
    #         passed += 1
    #     else:
    #         print(f"[✗] {message} → {response} [Expected: {expected}]")

    # accuracy = 100 * passed / len(test_data)
    # print(f"\nAccuracy: {accuracy:.2f}%")

# Run evaluation on start
# evaluate_chatbot()

# Run Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', port=5000, debug=True)
