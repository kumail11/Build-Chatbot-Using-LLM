from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


OPENAI_API_KEY = ''
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ.get("OPENAI_API_KEY")


conn = sqlite3.connect("store_responses.db", check_same_thread=False)
cur = conn.cursor()


cur.execute("""
    CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT)
""")

conn.commit()


# client = OpenAI(
#     api_key = os.environ.get('sk-proj-zyvt3vAqWx2dqYOXLGkPT3BlbkFJ74ehFpLtbywzivlKy8R0')
# )
# openai.api_key = "sk-proj-zyvt3vAqWx2dqYOXLGkPT3BlbkFJ74ehFpLtbywzivlKy8R0"
# openai.api_key = os.getenv('sk-proj-zyvt3vAqWx2dqYOXLGkPT3BlbkFJ74ehFpLtbywzivlKy8R0')


def compute_embeddings(text_data):
    vectorizer = TfidfVectorizer()
    tf_matrix = vectorizer.fit_transform(text_data)
    return tf_matrix.toarray()


# create some questions dictionary that can be stored in database in the vectors format..
questions_dict = {
    "what is machine learning": None,
    "what is data science": None,
    "why is ai important": None
}

question_text = list(questions_dict.keys())
question_embeddings = compute_embeddings(question_text)

questions_with_vectors = zip(question_text, question_embeddings)

cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY, question TEXT, vector TEXT)
""")


for question, vector in questions_with_vectors:
    vector_str = ' '.join(str(val) for val in vector)
    cur.execute("INSERT INTO questions (question, vector) VALUES (?, ?)", (question, vector_str))
    conn.commit()


@app.route("/natural_response", methods=['GET', 'POST'])
def natural_response():
    if request.method == 'POST':
        user_input = request.form['user_input']

        print(user_input)

        cur.execute("SELECT question from questions WHERE question = ?", (user_input,))
        question_match = cur.fetchone()

        if question_match:
            print("Question Match Found in the Database..")
            # Convert user input into vectors..
            user_input_vector = compute_embeddings([user_input])[0]

            print("Embeddings:\n", user_input_vector)

            # Now, convert this vector into text for open ai model..
            context = ''.join(str(val) for val in user_input_vector)
        
        else:
            print("No question found in the database..")
            context = user_input

        prompt = f"User Input Vector: {context}\nUser Input: {user_input}\nBot:"

        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150,
            top_p=1,
            stop=None
        )

        bot_response = response.choices[0].text.strip()

        cur.execute("INSERT INTO responses (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
        conn.commit()

        return jsonify({'bot_response': bot_response})

    
    return redirect(url_for("index"))





if __name__ == '__main__':
    app.run(debug=False)