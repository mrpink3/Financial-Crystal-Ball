import requests
import time
import csv
import os
from datetime import datetime

# Ayarları yap
API_KEY   = "API_KEY"  
START_YEAR = 2004
END_YEAR   = 2024
OUTPUT_FILE = "nyt_frontpage_archive_2004_2024.csv"

# CSV dosyasını hazırla
def init_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "year", "month", "pub_date", "headline", "snippet", "web_url", "print_page", "print_section"
        ])
        writer.writeheader()

init_csv(OUTPUT_FILE)

# Aylık olarak Archive API’yi çağır ve filtrele
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        print(f"Fetching Archive for {year}-{month:02d}...")
        base_url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
        full_url = f"{base_url}?api-key={API_KEY}"
        print(f"Full Request URL: {full_url}")  # Debug

        # Bu ay için veriyi çekene kadar tekrar dene
        while True:
            try:
                res = requests.get(full_url)
                if res.status_code == 429:
                    print(f"  ⏱ Rate limit yedi for {year}-{month:02d}, 60s bekleniyor ve retry ediliyor...")
                    time.sleep(60)
                    continue
                res.raise_for_status()
                data = res.json()
                break  # Başarılı çekim, çık
            except requests.HTTPError as http_err:
                print(f"  ⚠️ HTTP error for {year}-{month:02d}: {http_err}")
                time.sleep(5)
                break  # Başka HTTP hatasında tekrar denemek yerine atla
            except Exception as e:
                print(f"  ⚠️ Genel hata for {year}-{month:02d}: {e}")
                time.sleep(5)
                break  # Diğer hatalarda atla

        # Eğer data yoksa bu ay atlanmış demektir
        docs = data.get("response", {}).get("docs") or []
        # Sadece basılı ön sayfa (print_page == "1") ve print_section == "A"
        front_docs = [d for d in docs if d.get("print_page") == "1" and d.get("print_section") == "A"]

        # CSV’ye yaz
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "year", "month", "pub_date", "headline", "snippet", "web_url", "print_page", "print_section"
            ])
            for d in front_docs:
                writer.writerow({
                    "year":          year,
                    "month":         month,
                    "pub_date":      d.get("pub_date", ""),
                    "headline":      d.get("headline", {}).get("main", ""),
                    "snippet":       d.get("lead_paragraph", "") or d.get("snippet", ""),
                    "web_url":       d.get("web_url", ""),
                    "print_page":    d.get("print_page", ""),
                    "print_section": d.get("print_section", "")
                })

        print(f"  → {len(front_docs)} front-page articles (section A) saved.")
        # Rate limit uyumu için bekle

print("✅ All months from 2004 to 2024 processed for print_section A. Output:", OUTPUT_FILE)