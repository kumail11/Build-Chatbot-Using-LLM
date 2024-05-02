# ============================== Pinecone Implementation =========================

# from sklearn.metrics.pairwise import cosine_similarity
# from pinecone import Pinecone


# PINECONE_API_KEY = '1f5cb53c-d350-4c6c-8a26-a901582bb50d'
# # os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# pinecone = Pinecone(api_key=PINECONE_API_KEY)

# # Connect to the index (create if it doesn't exist)
# index_name = 'knowledge_index'
# dimension = 300
# spec = {"method": "hnsw", "ef_construction": 200, "M": 64}
# pinecone.create_index(index_name, dimension=dimension, spec=spec, metric='cosine')

# # Define the dataset (questions and answers)
# dataset = [
#     ("What is machine learning?", "Machine learning is a subset of artificial intelligence..."),
#     ("What is AI?", "Artificial intelligence (AI) is the simulation of human intelligence..."),
# ]

# # Convert text data to vectors using TF-IDF
# vectorizer = TfidfVectorizer()
# texts = [qa[0] + " " + qa[1] for qa in dataset]
# tfidf_matrix = vectorizer.fit_transform(texts)
# vectors = tfidf_matrix.toarray()

# # Insert vectors into the index
# pinecone.insert_items(index_name, vectors)

# # Query vectors (optional)
# query_vector = vectors[0]  # Example query vector
# top_k = 5  # Number of nearest neighbors to retrieve
# results = pinecone.query(index_name, query_vector, top_k=top_k)
# print("Top {} nearest neighbors:".format(top_k))
# for result in results:
#     print(result)

# # Disconnect from Pinecone server (optional)
# pinecone.delete_index(index_name)
# pinecone.close()


# def compute_embeddings(text_data):
#     vectorizer = TfidfVectorizer()
#     tf_matrix = vectorizer.fit_transform(text_data)
#     return tf_matrix.toarray()




# @app.route("/natural_response", methods=['GET', 'POST'])
# def natural_response():
#     if request.method == 'POST':
#         user_input = request.form['user_input']
#         print("User Input:", user_input)

#         user_input_vector = compute_embeddings([user_input])[0]
#         pinecone.insert_item(index_name, user_input_vector.tolist(), ids=[user_input])

#         # Query Pinecone index for nearest neighbors
#         results = pinecone.query(index_name, user_input_vector, top_k=3)
#         print("Pinecone Nearest Neighbors:", results)

#         # Retrieve text representations of nearest neighbors
#         nearest_neighbors = [pinecone.get_item(index_name, result.id) for result in results]

#         # Concatenate nearest neighbor texts with user input
#         context = user_input + "\n".join(nearest_neighbors)

#         # Generate response from OpenAI model
#         response = openai.completions.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=context,
#             max_tokens=150,
#             top_p=1,
#             stop=None
#         )

#         bot_response = response.choices[0].text.strip()

#         return jsonify({'bot_response': bot_response})

#     return redirect(url_for("index"))






















































# conn = sqlite3.connect("store_responses.db", check_same_thread=False)
# cur = conn.cursor()


# cur.execute("""
#     CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT)
# """)

# conn.commit()


# # client = OpenAI(
# #     api_key = os.environ.get('sk-proj-zyvt3vAqWx2dqYOXLGkPT3BlbkFJ74ehFpLtbywzivlKy8R0')
# # )
# # openai.api_key = "sk-proj-zyvt3vAqWx2dqYOXLGkPT3BlbkFJ74ehFpLtbywzivlKy8R0"
# # openai.api_key = os.getenv('sk-proj-zyvt3vAqWx2dqYOXLGkPT3BlbkFJ74ehFpLtbywzivlKy8R0')


# def compute_embeddings(text_data):
#     vectorizer = TfidfVectorizer()
#     tf_matrix = vectorizer.fit_transform(text_data)
#     return tf_matrix.toarray()


# # create some questions dictionary that can be stored in database in the vectors format..
# questions_dict = {
#     "what is machine learning": None,
#     "what is data science": None,
#     "why is ai important": None
# }

# question_text = list(questions_dict.keys())
# question_embeddings = compute_embeddings(question_text)

# questions_with_vectors = zip(question_text, question_embeddings)

# cur.execute("""
#     CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY, question TEXT, vector TEXT)
# """)


# for question, vector in questions_with_vectors:
#     vector_str = ' '.join(str(val) for val in vector)
#     cur.execute("INSERT INTO questions (question, vector) VALUES (?, ?)", (question, vector_str))
#     conn.commit()


# @app.route("/natural_response", methods=['GET', 'POST'])
# def natural_response():
#     if request.method == 'POST':
#         user_input = request.form['user_input']

#         print(user_input)

#         cur.execute("SELECT question from questions WHERE question = ?", (user_input,))
#         question_match = cur.fetchone()

#         if question_match:
#             print("Question Match Found in the Database..")
#             # Convert user input into vectors..
#             user_input_vector = compute_embeddings([user_input])[0]

#             print("Embeddings:\n", user_input_vector)

#             # Now, convert this vector into text for open ai model..
#             context = ''.join(str(val) for val in user_input_vector)
        
#         else:
#             print("No question found in the database..")
#             context = user_input

#         prompt = f"User Input Vector: {context}\nUser Input: {user_input}\nBot:"

#         response = openai.completions.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             max_tokens=150,
#             top_p=1,
#             stop=None
#         )

#         bot_response = response.choices[0].text.strip()

#         cur.execute("INSERT INTO responses (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
#         conn.commit()

#         return jsonify({'bot_response': bot_response})

    
#     return redirect(url_for("index"))
