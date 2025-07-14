import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
import time
from google.colab import files
import pandas as pd

# 2. Dataset'i yükle ve örnek al
df = pd.read_csv('nyt_frontpage_all_clean_csv.csv')
df_sample = df.sample(n=100, random_state=42).reset_index(drop=True)


# 3. GPT-2 model yüklemesi
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pipe = TextGenerationPipeline(model, tokenizer, device=0 if torch.cuda.is_available() else -1)

# 4. Hisse senedi adı
hissticker = input("Enter stock ticker (e.g., ASELS): ").upper()
sentiment_col = f"sentiment_{hissticker}"
df_sample[sentiment_col] = None

# 5. Sentiment fonksiyonu
def predict_sentiment(text, ticker):
    prompt = (f"News: {text}\n"
              f"Question: How will this news affect the stock price of {ticker}?\n"
              f"Answer with only one number: -1 (negative), 0 (neutral), 1 (positive).\nAnswer:")
    output = pipe(prompt, max_length=len(prompt.split()) + 10, do_sample=False, truncation=True)[0]["generated_text"]
    ans = output[len(prompt):].strip().split()[0]
    return ans if ans in ["-1", "0", "1"] else "0"

# 6. Ölçüm başlat
total_start = time.time()

for i in range(len(df_sample)):
    haber = str(df_sample.loc[i, "fulltext_clean"])[:200]
    start = time.time()
    result = predict_sentiment(haber, hissticker)
    elapsed = time.time() - start

    df_sample.at[i, sentiment_col] = result
    print(f"{i+1}/100 - Took {round(elapsed, 2)} seconds")

# 7. Toplam süre
total_elapsed = time.time() - total_start
print(f"\n✅ All 100 samples processed in {round(total_elapsed, 2)} seconds")

# 8. CSV olarak kaydet
df_sample.to_csv(f"sample_sentiment_{hissticker}.csv", index=False)
files.download(f"sample_sentiment_{hissticker}.csv")
