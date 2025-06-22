from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

import torch, scipy

def finbert_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = scipy.special.softmax(outputs.logits.numpy().squeeze())
    labels = [model.config.id2label[i] for i in range(len(scores))]
    best = labels[scores.argmax()]
    return {"positive": scores[labels.index("positive")],
            "negative": scores[labels.index("negative")],
            "neutral":  scores[labels.index("neutral")],
            "label":    best}

if __name__ == "__main__":
    text = "The company reported a significant increase in revenue this quarter."
    sentiment = finbert_sentiment(text)
    print(f"Sentiment for '{text}': {sentiment}")
    
    text = "The stock market is expected to crash soon."
    sentiment = finbert_sentiment(text)
    print(f"Sentiment for '{text}': {sentiment}")
    
    text = "The company's performance was stable with no major changes."
    sentiment = finbert_sentiment(text)
    print(f"Sentiment for '{text}': {sentiment}")