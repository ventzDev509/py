import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Yon lòt "Sèvo" ki pi fò
from sklearn.preprocessing import LabelEncoder

# 1. Simulation done ki soti nan tab TRACK ak LIKE ou yo
# Imajine sa se SELECT t.genre, t.duration, l.id FROM Track t LEFT JOIN Like l...
db_data = {
    'genre': ['Konpa', 'Rabòday', 'Zouk', 'Konpa', 'Rara', 'Konpa'],
    'duration': [320.5, 180.0, 240.2, 310.0, 200.0, 295.0], # soti nan Track.duration
    'liked': [1, 0, 1, 1, 0, 1] # 1 si gen yon entry nan tab Like, 0 si pa genyen
}

df = pd.DataFrame(db_data)

# 2. Preprocessing (Netwayaj)
le_genre = LabelEncoder()
df['genre_encoded'] = le_genre.fit_transform(df['genre'])

# 3. Chwazi Features (X) ak Target (y)
X = df[['genre_encoded', 'duration']]
y = df['liked']

# 4. Antrenman ak yon Random Forest (Pwofesyonèl pou done tabilè)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

print("✅ Modèl AI a aprann ak siksè sou Schema H-MIZIK la!")