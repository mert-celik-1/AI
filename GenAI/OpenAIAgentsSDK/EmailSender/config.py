"""Konfigürasyon değişkenleri."""
import os
from dotenv import load_dotenv

# Çevre değişkenlerini yükle
load_dotenv(override=True)

# OpenAI API anahtarı
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# SMTP ayarları
SMTP_HOST = 'localhost'
SMTP_PORT = 1025
EMAIL_FROM = 'gonderen@example.com'
EMAIL_TO = 'receiver@gmail.com'

# Model ayarları
MODEL = 'gpt-4o-mini'