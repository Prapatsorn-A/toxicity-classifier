from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load models and tokenizers
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Load Odd Layer model
odd_tokenizer, odd_model = load_model("model/odd_layer_model")

# Define label mapping
id2label = {0: "non-toxic", 1: "toxic"}

# Function to classify text
def classify_toxicity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return id2label[predicted_class]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = classify_toxicity(text, odd_tokenizer, odd_model)
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
