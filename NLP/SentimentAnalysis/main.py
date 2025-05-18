from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,pipeline

dataset = load_dataset("sst2")

print(f"Dataset yapısı : {dataset}")
print(f"Eğitim seti boyutu: {len(dataset['train'])}")
print(f"Örnek veri: {dataset['train'][0]}")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
result = classifier("I Really enjoyed this movie")
print(f"Sentiment analizi sonucu: {result}")

# 4. Tokenizasyon incelemesi - NLP'nin temel kavramı
text = "Let's understand how tokenization works in NLP models!"
tokens = tokenizer(text, return_tensors="pt")
print(f"Tokenize edilmiş girdi: {tokens}")
print(f"Token ID'leri: {tokens['input_ids']}")
print(f"Orijinal tokenler: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")

