"""
1. Word Embeddings Nedir ?

Word embeddings, kelimeleri bilgisayarın anlayabileceği sayı dizilerine (vektörlere) çevirme yöntemidir. Bu vektörler,
kelimelerin anlamını ve birbirine ne kadar benzediğini sayısal olarak temsil eder. Örneğin, "king" ve "queen" gibi benzer
kelimeler, uzayda birbirine yakın noktalara yerleştirilir. Bu sayede model, kelimeler arasında benzerlikleri ve ilişkileri öğrenebilir.
Tokenizasyon işleminden sonra embedding yapılır, yani kelimeler önce ayrılır, sonra sayılara çevrilir.

2. Embedding Türleri

Statik Embeddings:

Word2Vec: Google tarafından 2013'te tanıtıldı, context penceresi kullanır
GloVe: Stanford tarafından geliştirildi, global eş-bulunma istatistiklerini kullanır
FastText: Facebook tarafından geliştirildi, karakter n-gramlarını kullanarak OOV (sözlükte olmayan) kelimeleri ele alır

Contextual Embeddings (Bağlamsal Gömmeler):

BERT, RoBERTa, GPT gibi transformerların içindeki embedding katmanları
Aynı kelime, bağlama göre farklı vektörlere sahip olabilir
Statik embeddinglarden daha güçlüdür

"""

from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Temel kurulum
sentences = [
    "The king rules the kingdom.",
    "The queen loves her subjects.",
    "The man works hard every day.",
    "The woman reads many books.",
    "The child plays in the garden."
]

# 1. BERT Kullanarak Contextual Embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenizasyon
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Model ile embedding'leri al
with torch.no_grad():
    outputs = model(**encoded_input)
    embeddings = outputs.last_hidden_state

print(f"BERT Embeddings Şekli: {embeddings.shape}")
# [cümle sayısı, maksimum token uzunluğu, hidden size]

# 2. Özel kelimeler için embeddings çıkarma
target_words = ["king", "queen", "man", "woman", "child"]
word_embeddings = {}

for word in target_words:
    # Kelimeyi tokenize et
    inputs = tokenizer(word, return_tensors="pt")

    # Embedding al
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token'ı atlayıp, ilk gerçek token'ın embedding'ini al
    word_embeddings[word] = outputs.last_hidden_state[0, 1].numpy()

# 3. Kelimeler arası benzerlikler
words = list(word_embeddings.keys())
vectors = np.array([word_embeddings[word] for word in words])
similarities = cosine_similarity(vectors)

print("\nKelimeler Arası Benzerlikler:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:  # Tekrarları önle
            print(f"{word1} - {word2}: {similarities[i][j]:.4f}")


# 4. Vektör aritmetiği ile analojiler
def find_closest_word(target_vector):
    max_similarity = -1
    closest_word = None

    for word, vector in word_embeddings.items():
        similarity = cosine_similarity([target_vector], [vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            closest_word = word

    return closest_word, max_similarity


# Meşhur "king - man + woman = ?" analojisi
analogy_vector = word_embeddings["king"] - word_embeddings["man"] + word_embeddings["woman"]
result, score = find_closest_word(analogy_vector)

print(f"\nAnoloji: king - man + woman = {result} (benzerlik skoru: {score:.4f})")

# 5. Basit 2D görselleştirme (PCA ile)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

# Kelime etiketlerini ekle
for i, word in enumerate(words):
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=12)

plt.title("Word Embeddings 2D görselleştirmesi")
plt.grid(True)
plt.tight_layout()
#plt.show()


"""
Sentence Embeddings
Cümle embedding'leri, tüm cümleyi tek bir vektörle temsil etmek için kullanılır ve doküman benzerliği, arama, kümeleme 
gibi görevlerde çok faydalıdır. Örneğin RAG'in "Retrieval" kısmında sentence embeddings, "Generation" kısmında ise 
LLM içindeki word embeddings kullanılır.
"""