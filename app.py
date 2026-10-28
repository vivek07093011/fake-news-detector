from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained ML model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    news = data.get("news", "").strip()

    if not news:
        return jsonify({
            "error": "Please enter some news text."
        }), 400

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]

    probabilities = model.predict_proba(news_vector)[0]
    confidence = max(probabilities) * 100

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
