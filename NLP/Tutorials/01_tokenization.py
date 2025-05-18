"""
1. Tokenizasyon Nedir ?

Tokenizasyon metni bilgisayarın anlayabileceği parçalara ayırma işlemidir. Metin verilerini işlerken atılan ilk adımdır ve
sonraki tüm NLP işlemlerinin temelini oluşturur. Bir metin dizesini daha küçük birimler olan tokenlara bölme işlemidir.

Tokenlar şunlar olabilir :
- Kelimeler
- Alt kelimeler (subwords)
- Karakterler
- Noktalama işaretleri

"Hugging Face is amazing!" → ["Hugging", "Face", "is", "amazing", "!"]

2. Tokenizasyon Türleri

2.1 Kelime Tabanlı Tokenizasyon

En basit yaklaşım, metni boşluklara göre kelimelere ayırmaktır.
Örnek :
"""
text = "Hugging Face is amazing!"
simple_tokens = text.split()
print(f"Basit kelime tokenizasyonu: {simple_tokens}")
# Çıktı: ['Hugging', 'Face', 'is', 'amazing!'] -> Noktalama işaretlerini uygun şekilde alamaz

"""
2.2 Kelime + Noktalama Tokenizasyonu
Örnek :
"""
import re

def tokenize_with_punctuation(text):
    # Noktalama işaretlerinin önüne ve arkasına boşluk ekle
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    # Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

text = "Hugging Face is amazing!"
tokens = tokenize_with_punctuation(text)
print(f"Noktalama duyarlı tokenizasyon: {tokens}")
# Çıktı: ['Hugging', 'Face', 'is', 'amazing', '!'] -> Bileşik kelimeler,kısaltmalar ve özel durumlar için yetersiz kalabilir
# Örneğin "don't" kelimesi "don","","t" olarak ayrılırsa anlamsal bütünlüğü bozulabilir.

"""
2.3 Sub-word Tokenizasyon (Modern Yaklaşım)

Modern NLP'de en yaygın olarak kullanılan türdür. Kelimeleri anlamlı alt alt parçalara böler.

-BPE (Byte-Pair Encoding)
En sık görülen karakter çiftlerini birleştirerek tokenler oluşturur. Başlangıçta metin tek tek karakterlere bölünür. Ardından en çok
geçen çiftler adım adım birleştirilir. Bu sayede hem yaygın kelime parçaları hem de nadir kelimeler etkili şekilde temsil edilir. Dil modellerinin
hem kelime hem de kelime parçası seviyesinde esnek çalışmasını sağlar. GPT modelleri kullanır
Örnek:
"""
from transformers import GPT2Tokenizer

# GPT-2 BPE tokenizer kullanımı
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hugging Face is developing transformers tokenization!"
tokens = tokenizer.tokenize(text)
print(f"BPE tokenizasyonu (GPT-2): {tokens}")
# Örnek çıktı: ['Hug', 'ging', 'ĠFace', 'Ġis', 'Ġdeveloping', 'Ġtransform', 'ers', 'Ġtoken', 'ization', '!'] -> Ğ karakteri boşluğu temsil ediyor.


"""
-WordPiece
Bilinmeyen kelimeleri tanımlamak için kelimeleri olası en alt birimlere böler. BERT modeli kullanır. Küçük harflerle çalışır.
Parçalanmış kelimenin önünde ## işaretleri varsa bir önceki kelimenin devamı olduğuna işaret eder. Bu, decode işlemi sırasında boşluk eklenmemesi gerektiğini belirtir.
Örnek :
"""

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Unbelievable reactions were observed during the experiment."
tokens = tokenizer.tokenize(text)
print(f"WordPiece tokenizasyonu (BERT): {tokens}")
# Örnek çıktı:  ['unbelievable', 'reactions', 'were', 'observed', 'during', 'the', 'experiment', '.']


"""
SentencePiece
Dil bağımsız tokenizasyon sağlar ve boşlukları da token olarak ele alır. XLM, T5 gibi çok dilli modellerde kullanılır.
Boşlukları _ olarak işler. Dil bağımsız çalışır.
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

text = "Hugging Face is developing transformers!"
tokens = tokenizer.tokenize(text)
print(f"SentencePiece tokenizasyonu (XLNet): {tokens}")
# Örnek  çıktı : ['▁Hu', 'gging', '▁Face', '▁is', '▁developing', '▁transform', 'ers', '!']


"""
3. Hugging Face Tokenizer'ları Derinlemesine İnceleme
3.1 Tokenizer Yapısı ve Özellikleri
Hugging Face tokenizer'ları şu bileşenleri içerir:

Vocabulary: Token-ID eşleştirmeleri
Special tokens: [CLS], [SEP], [MASK] gibi özel tokenler
Normalization rules: Büyük-küçük harf dönüşümü, aksanlı karakterlerin işlenmesi
Pre-tokenization rules: Ön-tokenizasyon kuralları
Model-specific encoding: Model-spesifik kodlama kuralları
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenizer özelliklerini keşfetme
print(f"Vocabulary boyutu: {tokenizer.vocab_size}") # 30522
print(f"Özel tokenler: {tokenizer.all_special_tokens}") # ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
print(f"Padding token: {tokenizer.pad_token}") # PAD
print(f"Bilinmeyen token: {tokenizer.unk_token}") # UNK
print(f"Model max uzunluğu: {tokenizer.model_max_length}") # 512

# Sözlüğün bir kısmını görüntüleme
vocab_subset = dict(list(tokenizer.vocab.items())[:10])
print(f"Vocabulary'den örnek: {vocab_subset}") # {'[unused924]': 929, 'enclosed': 10837, 'credit': 4923, 'owe': 12533, 'newscast': 20306, 'hansen': 13328, 'gillespie': 21067, 'planting': 14685, 'spat': 14690, '##ity': 3012}


"""
3.2 Detaylı Tokenizasyon Süreci
Örnekler :
"""

from transformers import BertTokenizer

# BERT tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Örnek metinler
texts = [
    "Hugging Face is amazing!",
    "I love NLP and deep learning.",
    "Tokenization is a fundamental step in NLP.",
    "Python is great for machine learning.",
    "The quick brown fox jumps over the lazy dog."
]

# Her metin için tokenizasyon sürecini inceleme
for text in texts:
    print(f"\n\n{'=' * 50}")
    print(f"ORİJİNAL METİN: \"{text}\"")
    print(f"{'=' * 50}")

    # 1. Metni küçük harfe çevirme (BERT uncased için)
    normalized_text = text.lower()
    print(f"\n1. NORMALIZATION:")
    print(f"   {normalized_text}")

    # 2. Basit tokenizasyon (kelime düzeyinde)
    simple_tokens = normalized_text.split()
    print(f"\n2. BASİT KELİME TOKENIZASYONU:")
    print(f"   {simple_tokens}")

    # 3. BERT tokenizer ile tokenizasyon
    tokens = tokenizer.tokenize(text)
    print(f"\n3. BERT WORDPIECE TOKENIZASYONU:")
    print(f"   {tokens}")

    # 4. Token --> ID dönüşümü
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\n4. TOKEN ID'LERİ:")
    print(f"   {token_ids}")

    # 5. Tam encode işlemi
    encoding = tokenizer(text, return_tensors="pt")
    print(f"\n5. TAM ENCODING (special tokenlar dahil):")
    print(f"   input_ids: {encoding['input_ids'][0].tolist()}")
    print(f"   attention_mask: {encoding['attention_mask'][0].tolist()}")

    # 6. Special tokenların yerleştirilmesi
    special_tokens_added = tokenizer.build_inputs_with_special_tokens(token_ids)
    print(f"\n6. SPECIAL TOKEN'LAR EKLENMIŞ:")
    print(f"   {special_tokens_added}")
    print(f"   [CLS] token ID: {tokenizer.cls_token_id}")
    print(f"   [SEP] token ID: {tokenizer.sep_token_id}")

    # 7. ID --> Token geri dönüşümü
    decoded_tokens = tokenizer.convert_ids_to_tokens(special_tokens_added)
    print(f"\n7. ID'LERDEN TOKEN'LARA DÖNÜŞÜM:")
    print(f"   {decoded_tokens}")

    # 8. Tam decode işlemi
    decoded_text = tokenizer.decode(special_tokens_added)
    print(f"\n8. TAM DECODE:")
    print(f"   \"{decoded_text}\"")

    # 9. Özel token ID'leri olmadan decode
    decoded_text_clean = tokenizer.decode(special_tokens_added, skip_special_tokens=True)
    print(f"\n9. ÖZEL TOKEN'LAR OLMADAN DECODE:")
    print(f"   \"{decoded_text_clean}\"")

    # 10. Alt kelime (subword) tokenların analizi
    print(f"\n10. ALT KELİME TOKENLARIN ANALİZİ:")
    for token in tokens:
        if token.startswith("##"):
            print(f"   '{token}': Bir kelimenin devamı olan alt token")
        else:
            print(f"   '{token}': Kelime başlangıcı veya tam kelime")



"""
3.3 Farklı Metinlerin Tokenize Edilmesi ve Analizi
Örnekler :
"""

from transformers import AutoTokenizer
import pandas as pd

# Farklı model tokenizer'larını yükleme
tokenizers = {
    "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "RoBERTa": AutoTokenizer.from_pretrained("roberta-base"),
    "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
    "T5": AutoTokenizer.from_pretrained("t5-base"),
    "XLNet": AutoTokenizer.from_pretrained("xlnet-base-cased")
}

# Test metinleri
test_texts = [
    "Hugging Face transformers library is amazing!",  # Normal cümle
    "COVID-19 has affected the world economy.",  # Sayılar ve kısaltmalar
    "I don't want to walk 500 miles.",  # Apostroflu kelimeler
    "The email address is example@huggingface.co.",  # E-posta
    "This is a looooooong wooooooord.",  # Tekrarlanan harfler
    "https://huggingface.co is the website URL.",  # URL
    "Python3.8 and #hashtags are special tokens.",  # Özel karakterler
    "She said, 'Hello world!' with excitement.",  # Alıntılar
    "नमस्ते दुनिया! (Hello world in Hindi)",  # Farklı dil/alfabe
    "This costs $42.50 for 3 items."  # Para birimi ve sayılar
]

# Her tokenizer ve her metin için sonuçları karşılaştırma
results = []

for text_idx, text in enumerate(test_texts):
    for tokenizer_name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        results.append({
            "Text ID": text_idx + 1,
            "Text": text,
            "Tokenizer": tokenizer_name,
            "Tokens": tokens,
            "Token Count": len(tokens),
            "Has Special Characters": any(c in text for c in "!@#$%^&*()[]{}|\\;:'\",.<>/?"),
            "Has Numbers": any(c.isdigit() for c in text),
            "Avg Token Length": sum(len(t.replace("##", "").replace("▁", "")) for t in tokens) / len(
                tokens) if tokens else 0
        })

# Sonuçları DataFrame'e dönüştürme
df = pd.DataFrame(results)

# Tokenizer'lar arasındaki token sayısı farklarını analiz etme
pivot_counts = df.pivot_table(
    index=["Text ID", "Text"],
    columns="Tokenizer",
    values="Token Count",
    aggfunc="first"
)

print("\nTOKEN SAYILARI KARŞILAŞTIRMASI:")
print(pivot_counts)

# En çok ve en az tokena bölünen metinleri bulma
for tokenizer_name in tokenizers.keys():
    max_tokens_row = df[df["Tokenizer"] == tokenizer_name].loc[
        df[df["Tokenizer"] == tokenizer_name]["Token Count"].idxmax()]
    min_tokens_row = df[df["Tokenizer"] == tokenizer_name].loc[
        df[df["Tokenizer"] == tokenizer_name]["Token Count"].idxmin()]

    print(f"\n{tokenizer_name} için:")
    print(f"  En çok token ({max_tokens_row['Token Count']}): \"{max_tokens_row['Text']}\"")
    print(f"  Tokenler: {max_tokens_row['Tokens']}")
    print(f"  En az token ({min_tokens_row['Token Count']}): \"{min_tokens_row['Text']}\"")
    print(f"  Tokenler: {min_tokens_row['Tokens']}")

# Özel durumlar için ayrıntılı inceleme
print("\n\nÖZEL DURUMLARIN ANALİZİ:")

for text_idx, text in enumerate(test_texts):
    print(f"\n{'-' * 100}")
    print(f"TEXT {text_idx + 1}: \"{text}\"")

    for tokenizer_name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        print(f"\n{tokenizer_name} tokenizasyonu ({len(tokens)} token):")
        print(f"  {tokens}")

        # Özel analiz
        if "don't" in text and tokenizer_name == "BERT":
            print("  ÖZEL ANALİZ: BERT 'don't' kelimesini nasıl tokenize ediyor?")
            dont_indices = [i for i, t in enumerate(tokens) if "don" in t or "'" in t or "t" in t]
            print(f"  İlgili tokenler: {[tokens[i] for i in dont_indices]}")

        if "@" in text:
            print("  ÖZEL ANALİZ: Email adresi nasıl tokenize ediliyor?")
            at_index = text.find("@")
            relevant_tokens = [t for t in tokens if any(c in t for c in "@.")]
            print(f"  İlgili tokenler: {relevant_tokens}")

        if "looooooong" in text:
            print("  ÖZEL ANALİZ: Tekrarlanan harfler nasıl tokenize ediliyor?")
            long_tokens = [t for t in tokens if "loo" in t or "ooo" in t or "ong" in t]
            print(f"  İlgili tokenler: {long_tokens}")


"""
3.4 Tokenizasyon Sırasında Yaşanan Zorluklar ve Çözümleri
Örnek :
"""

from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenizasyon zorlukları
challenges = [
    {
        "category": "Bilinmeyen kelimeler",
        "example": "Supercalifragilisticexpialidocious is a long word.",
        "challenge": "Sözlükte olmayan kelimeler çok sayıda alt tokene bölünür."
    },
    {
        "category": "Teknik terimler",
        "example": "PyTorch and TensorFlow are deep learning frameworks.",
        "challenge": "Teknik terimler (PyTorch, TensorFlow) sık kullanılmayan alt tokenlere bölünebilir."
    },
    {
        "category": "Yazım hataları",
        "example": "I uesd the wrnog spellnig for words.",
        "challenge": "Yazım hataları tokenizasyonu beklenmedik şekilde etkileyebilir."
    },
    {
        "category": "Emoji ve semboller",
        "example": "I love NLP 😍 and programming 💻!",
        "challenge": "Emoji ve özel semboller beklenmedik şekilde tokenize edilebilir."
    },
    {
        "category": "Birleşik kelimeler",
        "example": "HandsOn and hands-on are compound words.",
        "challenge": "Farklı şekillerde yazılan birleşik kelimeler farklı tokenize edilebilir."
    },
    {
        "category": "Çok dilli içerik",
        "example": "English words and Türkçe kelimeler in the same text.",
        "challenge": "Farklı dillerdeki kelimeler karışık bir metinde tokenizasyon sorunlarına yol açabilir."
    }
]

# Her zorluk için tokenizasyon sonucunu inceleme
results = []

for challenge in challenges:
    tokens = tokenizer.tokenize(challenge["example"])
    token_ids = tokenizer.encode(challenge["example"])

    results.append({
        "Category": challenge["category"],
        "Example": challenge["example"],
        "Challenge": challenge["challenge"],
        "Tokens": tokens,
        "Token Count": len(tokens),
        "Tokens/Words Ratio": len(tokens) / len(challenge["example"].split())
    })

challenge_df = pd.DataFrame(results)
print("\nTOKENIZASYON ZORLUKLARI:")
for idx, row in challenge_df.iterrows():
    print(f"\n{'-' * 80}")
    print(f"KATEGORİ: {row['Category']}")
    print(f"ÖRNEK: \"{row['Example']}\"")
    print(f"ZORLUK: {row['Challenge']}")
    print(f"TOKENLER ({row['Token Count']}): {row['Tokens']}")
    print(f"TOKEN/KELİME ORANI: {row['Tokens/Words Ratio']:.2f}")

# Çözümler ve en iyi uygulamalar
print("\n\n" + "=" * 100)
print("TOKENIZASYON SORUNLARI İÇİN ÇÖZÜMLER VE EN İYİ UYGULAMALAR")
print("=" * 100)

solutions = [
    {
        "problem": "Alt kelime fragmantasyonu (Bir kelimenin çok sayıda tokene bölünmesi)",
        "solution": "Model seçimini görev için uygun yapın. Domain-specific modeller kullanın. Fine-tuning öncesi tokenizer'ı domain verilerinizle genişletin.",
        "example_code": """
# Tokenizer'ı domain verileriyle genişletme örneği
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
domain_vocab = ["supercalifragilisticexpialidocious", "pytorch", "tensorflow"]

# NOT: Gerçek uygulamada, tokenizer sınıfının özelliklerine bağlı olarak 
# bu işlem farklılık gösterebilir ve tam olarak desteklenmeyebilir.
"""
    },
    {
        "problem": "Maksimum token uzunluğu sınırları",
        "solution": "Uzun metinleri pencere yaklaşımıyla işleyin, örtüşen pencereler kullanın, hiyerarşik modeller deneyin.",
        "example_code": """
# Uzun metinleri pencereli yaklaşımla işleme
def process_long_text(text, tokenizer, max_length=512, stride=256):
    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]

    # Metin çok uzunsa pencereler oluştur
    if len(input_ids) > max_length:
        windows = []
        for i in range(0, len(input_ids), stride):
            end_idx = min(i + max_length, len(input_ids))
            windows.append(input_ids[i:end_idx])
            if end_idx == len(input_ids):
                break
        return windows
    else:
        return [input_ids]
"""
    },
    {
        "problem": "Model-spesifik tokenizasyon farklılıkları",
        "solution": "Her model için doğru tokenizer'ı kullanın. AutoTokenizer kullanımını tercih edin.",
        "example_code": """
# Her zaman model ile uyumlu tokenizer kullanın
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
# AutoTokenizer otomatik olarak model ile uyumlu tokenizer'ı seçer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
"""
    },
    {
        "problem": "Çoklu dil ve karakter seti sorunları",
        "solution": "Çok dilli modeller kullanın (XLM-RoBERTa, mBERT) veya SentencePiece tabanlı tokenizer'lar tercih edin.",
        "example_code": """
# Çok dilli model ve tokenizer kullanımı
from transformers import AutoTokenizer, AutoModel

# XLM-RoBERTa çok dilli bir modeldir
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Farklı dillerde metin örnekleri
texts = [
    "This is English text.",
    "Dies ist deutscher Text.",
    "これは日本語のテキストです。",
    "Это русский текст.",
    "هذا نص عربي."
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print()
"""
    }
]

for idx, solution in enumerate(solutions):
    print(f"\n{idx + 1}. PROBLEM: {solution['problem']}")
    print(f"   ÇÖZÜM: {solution['solution']}")
    print(f"   ÖRNEK KOD:")
    print(f"{solution['example_code']}")

# Tokenizasyon performans değerlendirmesi
print("\n\n" + "=" * 100)
print("TOKENIZASYON PERFORMANS DEĞERLENDİRMESİ")
print("=" * 100)

import time
import numpy as np

# Farklı uzunluklarda metinler oluşturma
text_sizes = [10, 50, 100, 200, 400]
texts = {}

np.random.seed(42)  # Tekrarlanabilirlik için
vocab = ["the", "a", "an", "in", "on", "with", "for", "and", "but", "or",
         "if", "because", "although", "while", "when", "where", "what", "who",
         "how", "why", "this", "that", "these", "those", "they", "we", "I", "you",
         "he", "she", "it", "be", "have", "do", "say", "go", "get", "make", "see",
         "know", "take", "come", "think", "look", "want", "give", "use", "find",
         "tell", "ask", "work", "seem", "feel", "try", "leave"]

for size in text_sizes:
    words = np.random.choice(vocab, size=size)
    texts[size] = " ".join(words)

# Farklı tokenizer'ların performansını ölçme
performance_results = []

for tokenizer_name, tokenizer in tokenizers.items():
    for size, text in texts.items():
        # Tokenizasyon süresi ölçümü
        start_time = time.time()
        tokens = tokenizer.tokenize(text)
        tokenize_time = time.time() - start_time

        # Encoding süresi ölçümü
        start_time = time.time()
        encoding = tokenizer.encode(text, return_tensors="pt")
        encode_time = time.time() - start_time

        performance_results.append({
            "Tokenizer": tokenizer_name,
            "Text Size (words)": size,
            "Token Count": len(tokens),
            "Tokenize Time (ms)": tokenize_time * 1000,
            "Encode Time (ms)": encode_time * 1000,
            "Tokens/Word Ratio": len(tokens) / size
        })

performance_df = pd.DataFrame(performance_results)

print("\nPERFORMANS KARŞILAŞTIRMASI:")
print(performance_df.sort_values(by=["Text Size (words)", "Tokenizer"]))

# Tokenizasyon oranları karşılaştırması
print("\nTOKEN/KELİME ORANI KARŞILAŞTIRMASI:")
ratio_pivot = performance_df.pivot_table(
    index="Text Size (words)",
    columns="Tokenizer",
    values="Tokens/Word Ratio",
    aggfunc="first"
)
print(ratio_pivot)

# Tokenizasyon süreleri karşılaştırması
print("\nTOKENIZASYON SÜRELERİ KARŞILAŞTIRMASI (ms):")
time_pivot = performance_df.pivot_table(
    index="Text Size (words)",
    columns="Tokenizer",
    values="Tokenize Time (ms)",
    aggfunc="first"
)
print(time_pivot)