import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Log dosyasının bulunduğu yolu belirtme
log_file_path = 'access.log'

# Log dosyasını okuma
with open(log_file_path, 'r') as file:
    logs = file.readlines()

# Her satırı ayrıştırmak için bir düzenli ifade deseni
log_pattern = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+)'  # IP adresi
    r' - - '  # Sabit metin
    r'\[(?P<timestamp>[^\]]+)\] '  # Zaman damgası
    r'"(?P<method>\w+) '  # HTTP methodu
    r'(?P<url>[^\s]+)'  # URL
    r' [^"]+"'  # Protokol ve diğer sabit metinler
)

# Ayrıştırılmış verileri saklamak için bir liste
parsed_logs = []

# Her log satırını ayrıştırma
for log in logs:
    match = log_pattern.match(log)
    if match:
        parsed_logs.append(match.groupdict())

# Ayrıştırılmış log verilerini DataFrame'e dönüştürme
df = pd.DataFrame(parsed_logs)

# DataFrame'in boş olup olmadığını kontrol etme
if df.empty:
    print("Log verileri düzgün ayrıştırılamadı. Lütfen log dosyasını ve düzenli ifade desenini kontrol edin.")
    exit(1)

# İlk 5 kaydı gösterme
print("İlk 5 kayıt:")
print(df.head())

# Her log kaydını birleştirerek tek bir string oluşturma
df['combined'] = df.apply(lambda row: f"{row['ip']} {row['timestamp']} {row['method']} {row['url']}", axis=1)

# Birleştirilmiş log kayıtlarını TF-IDF vektörlerine dönüştürme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# TF-IDF matrisinin boyutlarını kontrol etme
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# TF-IDF matrisini numpy array'e dönüştürme
tfidf_vectors = tfidf_matrix.toarray().astype('float32')

# FAISS index oluşturma (L2 mesafe metriği)
index = faiss.IndexFlatL2(tfidf_vectors.shape[1])

# Vektörleri FAISS'e yükleme
index.add(tfidf_vectors)

# FAISS'deki toplam vektör sayısını kontrol etme
print("Toplam vektör sayısı:", index.ntotal)

def find_relevant_logs(user_query):
    # Tarih arama
    date_match = re.search(r'(\d{1,2}) (\w+) (\d{4})', user_query)
    if date_match:
        day, month, year = date_match.groups()
        # Ay adlarını kısaltma olarak almak için datetime modülünü kullanma
        from datetime import datetime
        month_number = datetime.strptime(month, "%B").strftime("%b")
        search_date = f"{day}/{month_number}/{year}"

        # Tarihe göre filtreleme
        filtered_logs = df[df['timestamp'].str.contains(search_date)]
    else:
        # Tarih belirtilmemişse anahtar kelimeye göre filtreleme
        keywords = user_query.split()
        filtered_logs = df[df['url'].str.contains('|'.join(keywords), case=False, na=False)]

    # Eğer uygun bir sonuç bulunamazsa
    if filtered_logs.empty:
        return None

    # Sonuçları toplama
    log_data = []
    for _, log_entry in filtered_logs.iterrows():
        log_data.append({
            'timestamp': log_entry['timestamp'],
            'method': log_entry['method'],
            'url': log_entry['url'],
            'ip': log_entry['ip']
        })
    return log_data

# Model ve tokenizer'ı yükleme
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Pad token tanımlama
tokenizer.pad_token = tokenizer.eos_token

def generate_answer(log_data, user_query):
    # Log kayıtlarını bir metin stringi haline getirme
    log_text = "\n".join([f"{log['ip']} - {log['timestamp']} - {log['method']} {log['url']}" for log in log_data])

    # Kullanıcının sorusunu ve ilgili log kayıtlarını birleştirme
    input_text = f"Kullanıcı sorusu: {user_query}\n\nRelevant log records:\n{log_text}\n\n"

    # Model ile yanıt oluşturma
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Yanıt üretimi için max_new_tokens kullanımı
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=10,  # Üretilecek yeni token sayısını düşürme
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.5,  # Daha düşük sıcaklık kullanarak rastgeleliği azaltma
        do_sample=False  # Sampling yerine deterministik bir çıktı elde etme
    )

    # Yanıtı decode etme
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Log kayıtlarının sonuna kadar yanıtı döndürme
    return answer

# Kullanıcı sorusuna dayalı yanıt oluşturmayı başlatma
def answer_user_query():
    while True:
        user_query = input("Please enter your query (or type 'exit' to quit): ")

        if user_query.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        log_data = find_relevant_logs(user_query)

        # Eğer log_data boşsa veya None dönerse, kullanıcıya bir mesaj gösterme
        if not log_data:
            print("No relevant log entries found. Please try a different query.")
            continue

        answer = generate_answer(log_data, user_query)
        print("Generated Answer:")
        print(answer)

# Kullanıcı sorusuna dayalı yanıt oluşturmayı başlatma
answer_user_query()
