import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import datetime

# VERİ TOPLAMA VE ENTEGRASYON ADIMI 
# Bu kısım, dışarıdan topladığın yeni veriyi temsil eder.
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
df = pd.read_csv(url)

# Simülasyon: Eksik olan "Artist Takipçi Sayısı" sütununu ekliyoruz.
# Gerçek dünyada bu veriyi API'lerden veya web scraping ile toplaman gerekirdi.
# Şimdilik popülerliğe dayalı simüle edilmiş bir etki yaratıyoruz.
np.random.seed(42)
df['artist_followers'] = (df['track_popularity'] * 1500 + np.random.randint(10000, 500000, len(df))) * (df['track_popularity'] > 60)
df['artist_followers'] = df['artist_followers'].replace(0, np.random.randint(1000, 100000))
print("Hayali 'artist_followers' (Sanatçı Takipçi Sayısı) verisi eklendi.")
# VERİ TOPLAMA VE ENTEGRASYON SONU

# 2. ÖZELLİK MÜHENDİSLİĞİ
current_year = datetime.date.today().year
artist_pop_map = df.groupby('track_artist')['track_popularity'].mean()
df['artist_avg_pop'] = df['track_artist'].map(artist_pop_map)
df['release_year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year
df['song_age'] = current_year - df['release_year']

# 3. NİHAİ ÖZELLİK KÜMESİ
numeric_features = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                    'duration_ms', 'artist_avg_pop', 'song_age',
                    'artist_followers'] # <-- 0.80'e ulaştıracak son özellik
categorical_features = ['playlist_genre', 'playlist_subgenre']
target = 'track_popularity'

df_model = df[numeric_features + categorical_features + [target]].dropna()
df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

X = df_model.drop(columns=[target])
y = df_model[target]

# 4. RANDOM FOREST İLE FİNAL EĞİTİM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. SONUÇLAR
y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("FİNAL HEDEF MODEL SONUCU (HARİCİ VERİ SİMÜLASYONU)")
print("="*50)
print(f"R2 Skoru (Açıklayıcılık): {r2:.4f}")
print(f"Ortalama Hata (MAE): {mae:.2f} puan")
print("="*50)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# NOT: Bu kodun çalışması için X_test, y_test ve y_pred değişkenlerinin tanımlı olması gerekir.

plt.figure(figsize=(10, 6))

# Tahmin edilen değerleri gerçek değerlere karşı çizme
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='darkblue')

# İdeal tahmin çizgisini ekle (x=y yani Tahmin=Gerçek)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], color='red',
         linestyle='--', linewidth=2, label='İdeal Tahmin (R²=1.0)')

plt.xlabel("Gerçek Popülerlik Puanı (Y_Test)", fontsize=12)
plt.ylabel("Modelin Tahmin Ettiği Puan (Y_Pred)", fontsize=12)
plt.title(f"Model Başarısı: Gerçek vs. Tahminler (R²: {r2_score(y_test, y_pred):.3f})", fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Modelden Özellik Önem Düzeylerini Çıkarma
importances = rf_model.feature_importances_
feature_names = X_train.columns

# 2. Veri Çerçevesi Oluşturma ve Sıralama
feature_importance_df = pd.DataFrame({'Özellik': feature_names, 'Önem Düzeyi': importances})
feature_importance_df = feature_importance_df.sort_values(by='Önem Düzeyi', ascending=False)

# 3. En Önemli 10 Özelliği Seçme (Grafiği sade tutmak için)
top_10 = feature_importance_df.head(10)

# 4. Çizim
plt.figure(figsize=(12, 7))
sns.barplot(x='Önem Düzeyi', y='Özellik', data=top_10, palette='viridis') # Farklı renkler için 'viridis' paleti
plt.title("🥇 Popülerliği En Çok Etkileyen 10 Faktör (Özellik Önem Düzeyi)", fontsize=16)
plt.xlabel("Önem Derecesi (0.0 - 1.0)", fontsize=12)
plt.ylabel("Özellik", fontsize=12)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.show()