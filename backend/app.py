from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load the pre-trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    post_content = data["text"]

    inputs = tokenizer(post_content, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

    # Map prediction to sentiment/crisis level
    sentiment_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
    crisis_level = 1 if sentiment_map[prediction] == "Negative" else 0

    return jsonify({"sentiment": sentiment_map[prediction], "crisis": crisis_level})


if __name__ == "__main__":
    app.run(debug=True)
