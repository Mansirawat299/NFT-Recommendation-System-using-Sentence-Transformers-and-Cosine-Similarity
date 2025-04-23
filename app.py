from flask import Flask, request, render_template
import pickle
from sklearn.metrics.pairwise import cosine_similarity # type: ignore


app = Flask(__name__)

with open('recommendation_system.pkl', 'rb') as file:
    components = pickle.load(file)

model = components['model']
nft_embeddings = components['nft_embeddings']
df = components['df']

# Recommendation function
def get_recommendations(query, embeddings, df, top_n=5):
    query_embedding = model.encode([query])  # Encode the query
    similarities = cosine_similarity(query_embedding, embeddings)  # Compute cosine similarity
    top_indices = similarities[0].argsort()[-top_n:][::-1]  # Get top-N indices
    return df.iloc[top_indices]  # Return top-N recommendations

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']  # Get the user's search query
        recommendations = get_recommendations(query, nft_embeddings, df, top_n=5)
        return render_template('index.html', query=query, recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)