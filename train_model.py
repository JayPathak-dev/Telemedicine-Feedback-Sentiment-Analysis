import pandas as pd
import numpy as np
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# ==========================================
# 0. SETUP NLP TOOLS
# ==========================================
print("Downloading NLTK resources...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==========================================
# 1. CLEAN TEXT FUNCTION (NEGATION SAFE)
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()

    important_negations = {"not", "no", "never"}

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if (word not in stop_words) or (word in important_negations)
    ]

    return " ".join(cleaned_words)

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading dataset...")
file_name = "dataSet.xls"
df = pd.read_excel(file_name, skiprows=6)

pos_texts = df['Liked'].dropna().tolist()
neg_texts_raw = df['Disliked'].dropna().tolist()

# ==========================================
# 3. SMART FILTERING (REMOVE LABEL NOISE)
# ==========================================
print("Cleaning dataset with smart filtering...")

negative_keywords = {
    "dirty", "rude", "poor", "bad", "terrible",
    "disappointed", "worst", "unprofessional",
    "unhelpful", "horrible"
}

positive_keywords = {
    "excellent", "great", "satisfied", "happy",
    "good", "professional", "caring",
    "friendly", "outstanding"
}

pos_clean = []
neg_clean = []

# Process Positive Column
for text in pos_texts:
    cleaned = clean_text(text)
    words = set(cleaned.split())

    if len(cleaned.split()) >= 4 and not words.intersection(negative_keywords):
        pos_clean.append(cleaned)

# Process Negative Column
for text in neg_texts_raw:
    cleaned = clean_text(text)
    words = set(cleaned.split())

    if len(cleaned.split()) >= 4 and not words.intersection(positive_keywords):
        neg_clean.append(cleaned)

print("After smart filtering:")
print("Positive:", len(pos_clean))
print("Negative:", len(neg_clean))

# ==========================================
# 4. BALANCE DATASET
# ==========================================
min_size = min(len(pos_clean), len(neg_clean))

pos_clean = pos_clean[:min_size]
neg_clean = neg_clean[:min_size]

print("After balancing:")
print("Positive:", len(pos_clean))
print("Negative:", len(neg_clean))

# ==========================================
# 5. CREATE DATAFRAME
# ==========================================
df_pos = pd.DataFrame({'text': pos_clean, 'label': 1})
df_neg = pd.DataFrame({'text': neg_clean, 'label': 0})

data = pd.concat([df_pos, df_neg], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total training samples:", len(data))

# ==========================================
# 6. TOKENIZATION
# ==========================================
MAX_VOCAB_SIZE = 8000
MAX_SEQUENCE_LENGTH = 120

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])

sequences = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding='post',
    truncating='post'
)

y = data['label'].values

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 7. BUILD MODEL
# ==========================================
print("Building Model...")

model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ==========================================
# 8. TRAIN MODEL
# ==========================================
print("Training Model...")

model.fit(
    X_train,
    y_train,
    epochs=6,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# ==========================================
# 9. SAVE MODEL & TOKENIZER
# ==========================================
print("Saving model and tokenizer...")

model.save("sentiment_model.h5")

with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("SUCCESS! Smart-filtered model saved.")
