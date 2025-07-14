import pandas as pd
from collections import Counter

# 1️⃣ CSV’yi oku
df = pd.read_csv("nyt_frontpage_finbert_only.csv")

# 2️⃣ Ortalama skor
mean_scores = df.groupby("date")["finbert_numeric"].mean().reset_index()
mean_scores.rename(columns={"finbert_numeric": "mean_score"}, inplace=True)

# 3️⃣ Majority Vote (mod)
def majority_vote(group):
    return Counter(group).most_common(1)[0][0]

majority_scores = df.groupby("date")["finbert_numeric"].agg(majority_vote).reset_index()
majority_scores.rename(columns={"finbert_numeric": "majority_vote_score"}, inplace=True)

# 4️⃣ Medyan
median_scores = df.groupby("date")["finbert_numeric"].median().reset_index()
median_scores.rename(columns={"finbert_numeric": "median_score"}, inplace=True)

# 5️⃣ Ortalama skoru sınıflandır (-1, 0, 1)
def classify_mean(x, threshold=0.3):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

mean_scores["mean_class"] = mean_scores["mean_score"].apply(classify_mean)

# 6️⃣ Birleştir ve Kaydet
final_df = mean_scores.merge(majority_scores, on="date").merge(median_scores, on="date")
final_df.to_csv("daily_finbert_aggregated_with_class.csv", index=False)

print("✅ Günlük FinBERT skorları threshold’lu şekilde başarıyla kaydedildi.")
