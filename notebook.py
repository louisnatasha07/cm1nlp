import os
import re
import string
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 160)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. BACA DATA
file_path = 'E:/SEM 6/NLP/CM1/spam.csv'

print("Membaca dataset...")
df = pd.read_csv(file_path, encoding='latin-1')

# Ambil kolom utama SMS Spam Collection
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. CEK DATA AWAL
print("\n=== 5 Data Teratas ===")
print(df.head())

print("\n=== Info Data ===")
print(df.info())

print("\n=== Nama Kolom ===")
print(df.columns.tolist())

# 3. EDA
print('Shape dataset:', df.shape)
print('\nInfo dataset:')
df.info()

print('\nMissing values:')
print(df.isna().sum())

print('\nDuplikasi data:', df.duplicated().sum())

print('\nDistribusi label:')
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True).mul(100).round(2).rename('percentage'))

print(df.sample(10, random_state=SEED))

# Analisis panjang teks
df['char_len'] = df['message'].astype(str).apply(len)
df['word_len'] = df['message'].astype(str).apply(lambda x: len(x.split()))

print(df.groupby('label')[['char_len', 'word_len']].describe())

plt.figure(figsize=(6,4))
df['label'].value_counts().plot(kind='bar')
plt.title('Distribusi Label')
plt.xlabel('Label')
plt.ylabel('Jumlah Data')
plt.show()

plt.figure(figsize=(7,4))
for label in df['label'].unique():
    df[df['label'] == label]['word_len'].plot(kind='hist', alpha=0.5, bins=30, label=label)
plt.title('Distribusi Panjang Pesan Berdasarkan Label')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()

# 4. Preprocessing Text
def clean_text(text):
    text = str(text).lower()                      
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)  
    text = re.sub(r'\d+', ' number ', text)          
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

label_map = {'ham': 0, 'spam': 1}
df['label_encoded'] = df['label'].map(label_map)
df['clean_message'] = df['message'].apply(clean_text)

print(df[['message', 'clean_message', 'label', 'label_encoded']].head(10))

# 5. Train-test split
X_raw = df['message']
X_clean = df['clean_message']
y = df['label_encoded']

X_train_raw, X_test_raw, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_raw,
    X_clean,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print('Jumlah data train:', len(X_train_text))
print('Jumlah data test :', len(X_test_text))
print('\nDistribusi train:')
print(y_train.value_counts(normalize=True))
print('\nDistribusi test:')
print(y_test.value_counts(normalize=True))

# 6. Helper Evaluasi
results = []

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print(f'=== {name} ===')
    print('Accuracy :', round(acc, 4))
    print('Precision:', round(precision, 4))
    print('Recall   :', round(recall, 4))
    print('F1-score :', round(f1, 4))
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=['ham', 'spam'], digits=4))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    results.append({
        'Method': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

def show_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['ham', 'spam'])
    disp.plot()
    plt.title(title)
    plt.show()
    
# 7.1 Bag of Words + Multinomial Naive Bayes
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)

bow_model = MultinomialNB()
bow_model.fit(X_train_bow, y_train)

pred_bow = bow_model.predict(X_test_bow)
evaluate_model('Traditional - BoW + MultinomialNB', y_test, pred_bow)

# 7.2 TF-IDF + Linear SVM
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

tfidf_model = LinearSVC(class_weight='balanced', random_state=SEED)
tfidf_model.fit(X_train_tfidf, y_train)

pred_tfidf = tfidf_model.predict(X_test_tfidf)
evaluate_model('Traditional - TF-IDF + LinearSVC', y_test, pred_tfidf)

# 7.3 N-gram + Logistic Regression
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
X_train_ngram = ngram_vectorizer.fit_transform(X_train_text)
X_test_ngram = ngram_vectorizer.transform(X_test_text)

ngram_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
ngram_model.fit(X_train_ngram, y_train)

pred_ngram = ngram_model.predict(X_test_ngram)
evaluate_model('Traditional - N-gram(1,2) + LogisticRegression', y_test, pred_ngram)

# 8. Static Embedding - FastText
from gensim.models import FastText

tokenized_train = X_train_text.apply(str.split).tolist()
tokenized_test = X_test_text.apply(str.split).tolist()

EMBEDDING_SIZE = 100

fasttext_model = FastText(
    sentences=tokenized_train,
    vector_size=EMBEDDING_SIZE,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=20,
    seed=SEED
)

def get_document_vector(tokens, model, vector_size=100):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

X_train_fasttext = np.array([
    get_document_vector(tokens, fasttext_model, EMBEDDING_SIZE)
    for tokens in tokenized_train
])
X_test_fasttext = np.array([
    get_document_vector(tokens, fasttext_model, EMBEDDING_SIZE)
    for tokens in tokenized_test
])

print('Shape FastText train:', X_train_fasttext.shape)
print('Shape FastText test :', X_test_fasttext.shape)

fasttext_clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
fasttext_clf.fit(X_train_fasttext, y_train)

pred_fasttext = fasttext_clf.predict(X_test_fasttext)
evaluate_model('Static Embedding - FastText + LogisticRegression', y_test, pred_fasttext)

# 9. Contextual Embedding - BERT Fine Tuning
# Jika dijalankan di Kaggle/Colab dan library belum tersedia, aktifkan baris ini:
# !pip install -q transformers datasets evaluate accelerate

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

BERT_MODEL_NAME = 'bert-base-uncased'

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

train_df = pd.DataFrame({
    'text': X_train_raw.tolist(),
    'label': y_train.tolist()
})

test_df = pd.DataFrame({
    'text': X_test_raw.tolist(),
    'label': y_test.tolist()
})

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_function(batch):
    return bert_tokenizer(
        batch['text'],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir='./bert_sms_spam_results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_steps=50,
    report_to='none',
    seed=SEED
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

bert_eval = trainer.evaluate()
print(bert_eval)

bert_predictions = trainer.predict(test_dataset)
pred_bert = np.argmax(bert_predictions.predictions, axis=-1)

evaluate_model('Contextual Embedding - BERT Fine-Tuned', y_test, pred_bert)
show_confusion_matrix(y_test, pred_bert, 'Confusion Matrix - BERT Fine-Tuned')

# 10. Final Comparison
results_df = pd.DataFrame(results).sort_values(by='F1-score', ascending=False).reset_index(drop=True)
print(results_df)

plt.figure(figsize=(10, 5))
plt.bar(results_df['Method'], results_df['F1-score'])
plt.xticks(rotation=35, ha='right')
plt.ylabel('Weighted F1-score')
plt.title('Perbandingan Performa Representasi dan Model')
plt.tight_layout()
plt.show()

# 11. Error Analysis
comparison_df = pd.DataFrame({
    'message': X_test_raw.tolist(),
    'clean_message': X_test_text.tolist(),
    'actual': y_test.tolist(),
    'pred_bow': pred_bow,
    'pred_tfidf': pred_tfidf,
    'pred_ngram': pred_ngram,
    'pred_fasttext': pred_fasttext,
    'pred_bert': pred_bert
})

label_inv = {0: 'ham', 1: 'spam'}
for col in ['actual', 'pred_bow', 'pred_tfidf', 'pred_ngram', 'pred_fasttext', 'pred_bert']:
    comparison_df[col] = comparison_df[col].map(label_inv)

print(comparison_df.head())

# Contoh error dari BERT fine-tuned
bert_errors = comparison_df[comparison_df['actual'] != comparison_df['pred_bert']]
print('Jumlah error BERT:', len(bert_errors))
print(bert_errors.head(10))
# ============================================================
# 12. DEMO PREDIKSI MODEL UNTUK VIDEO
# ============================================================
# Bagian ini digunakan untuk mendemonstrasikan bahwa model yang sudah dilatih
# dapat menerima input pesan baru dan menghasilkan prediksi ham/spam.

label_output = {0: 'ham', 1: 'spam'}

def predict_text_all_models(text):
    print('\n' + '=' * 80)
    print('INPUT PESAN:')
    print(text)
    print('=' * 80)

    # Preprocessing untuk model traditional dan FastText
    clean = clean_text(text)
    print('Hasil preprocessing:', clean)

    # 1. BoW
    bow_vec = bow_vectorizer.transform([clean])
    pred_bow_demo = bow_model.predict(bow_vec)[0]

    # 2. TF-IDF
    tfidf_vec = tfidf_vectorizer.transform([clean])
    pred_tfidf_demo = tfidf_model.predict(tfidf_vec)[0]

    # 3. N-gram
    ngram_vec = ngram_vectorizer.transform([clean])
    pred_ngram_demo = ngram_model.predict(ngram_vec)[0]

    # 4. FastText
    tokens = clean.split()
    fasttext_vec = get_document_vector(tokens, fasttext_model, EMBEDDING_SIZE)
    pred_fasttext_demo = fasttext_clf.predict([fasttext_vec])[0]

    # 5. BERT fine-tuned
    bert_model.eval()
    inputs = bert_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )

    # Jika model berada di GPU, input juga dipindahkan ke GPU
    model_device = next(bert_model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    pred_bert_demo = outputs.logits.argmax(dim=-1).item()

    print('\nHASIL PREDIKSI SEMUA MODEL:')
    print('BoW + MultinomialNB                 :', label_output[pred_bow_demo])
    print('TF-IDF + LinearSVC                  :', label_output[pred_tfidf_demo])
    print('N-gram(1,2) + LogisticRegression    :', label_output[pred_ngram_demo])
    print('FastText + LogisticRegression       :', label_output[pred_fasttext_demo])
    print('BERT Fine-Tuned                     :', label_output[pred_bert_demo])

    return {
        'input': text,
        'clean_text': clean,
        'BoW': label_output[pred_bow_demo],
        'TF-IDF': label_output[pred_tfidf_demo],
        'N-gram': label_output[pred_ngram_demo],
        'FastText': label_output[pred_fasttext_demo],
        'BERT Fine-Tuned': label_output[pred_bert_demo]
    }


print("\n================ DEMO INTERAKTIF =================\n")

while True:
    text = input("Masukkan pesan (ketik 'exit' untuk keluar): ")

    if text.lower() == 'exit':
        print("Demo selesai.")
        break

    print("\nINPUT:", text)

    # Preprocess
    clean = clean_text(text)
    print("Hasil preprocessing:", clean)

    # BoW
    bow_vec = bow_vectorizer.transform([clean])
    pred_bow = bow_model.predict(bow_vec)[0]

    # TF-IDF
    tfidf_vec = tfidf_vectorizer.transform([clean])
    pred_tfidf = tfidf_model.predict(tfidf_vec)[0]

    # N-gram
    ngram_vec = ngram_vectorizer.transform([clean])
    pred_ngram = ngram_model.predict(ngram_vec)[0]

    # FastText
    tokens = clean.split()
    vec = get_document_vector(tokens, fasttext_model)
    pred_fasttext = fasttext_clf.predict([vec])[0]

    # BERT
    import torch
    bert_model.eval()

    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    pred_bert = outputs.logits.argmax().item()

    label_map = {0: 'ham', 1: 'spam'}

    print("\nHASIL PREDIKSI:")
    print("BoW:", label_map[pred_bow])
    print("TF-IDF:", label_map[pred_tfidf])
    print("N-gram:", label_map[pred_ngram])
    print("FastText:", label_map[pred_fasttext])
    print("BERT:", label_map[pred_bert])

    print("\n----------------------------------------\n")