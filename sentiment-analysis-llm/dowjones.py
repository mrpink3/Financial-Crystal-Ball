import os
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# Model
base_model = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Hedef varlık listesi
tickers = [
    "Dow Jones"
]



# Veri
df = pd.read_csv("nyt_frontpage_all_clean_csv.csv")

# Tahmin fonksiyonu
def predict_sentiment(text, ticker):
    prompt = (
        f"<|system|>\nYou are a helpful financial assistant.\n"
        f"<|user|>\nWhat is the sentiment of the following financial news for {ticker}?\n"
        f"Choose one of: positive, neutral, negative.\n"
        f"Respond with one word only.\n\n"
        f"{text}\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=5)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_part = decoded.split("<|assistant|>")[-1].lower().strip()
    matches = re.findall(r"\b(positive|neutral|negative)\b", assistant_part)
    return matches[0] if matches else "unknown"

# Klasör oluştur
os.makedirs("sentiment_outputs", exist_ok=True)

# Her varlık için işlem
for ticker in tickers:
    print(f"\n🚀 {ticker.upper()} işleniyor...\n")
    sentiment_col = f"sentiment_{ticker.replace(' ', '_')}"
    df_copy = df.copy()
    df_copy[sentiment_col] = None

    for i in range(len(df_copy)):
        text = str(df_copy.loc[i, "fulltext_clean"])
        sentiment = predict_sentiment(text, ticker)
        df_copy.at[i, sentiment_col] = sentiment
        print(f"[{i+1}/{len(df_copy)}] {sentiment.upper()}")

    # CSV olarak kaydet
    clean_name = ticker.lower().replace(" ", "_").replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"sentiment_outputs/{clean_name}_sentiment_{timestamp}.csv"
    df_copy.to_csv(output_path, index=False)
    print(f"✅ {ticker} tamamlandı. Kaydedildi: {output_path}")

