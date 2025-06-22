from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

import torch, scipy
import pandas as pd

def finbert_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = scipy.special.softmax(outputs.logits.numpy().squeeze())
    labels = [model.config.id2label[i] for i in range(len(scores))]
    best = labels[scores.argmax()]
    """return {"positive": scores[labels.index("positive")],
            "negative": scores[labels.index("negative")],
            "neutral":  scores[labels.index("neutral")],
            "label":    best}"""
    return {"label": best}

if __name__ == "__main__":

    data = pd.read_csv("data/movement_dataset_with_total.csv")

    for i, row in data.iterrows():
        text = row["news_text"]
        sentiment = finbert_sentiment(text)
        # Apply a lambda function to set the sentiment label
        # if posivive 1 if neutral 0 if negative -1
        if sentiment["label"] == "positive":
            data.at[i, "sentiment"] = 1
        elif sentiment["label"] == "neutral":
            data.at[i, "sentiment"] = 0
        elif sentiment["label"] == "negative":
            data.at[i, "sentiment"] = -1
        else:
            data.at[i, "sentiment"] = None

    total = len(data)
    correct = len(data[data["sentiment"] == data["total_movement"]])
    accuracy = correct / total * 100
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")
    