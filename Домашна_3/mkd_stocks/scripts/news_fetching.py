import os
import requests
import html
import pdfplumber
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import time
from googletrans import Translator
from unidecode import unidecode
from textblob import TextBlob
from transformers import pipeline

# Постави ја патеката до tesseract.exe ако не е автоматски откриен
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_URL = "https://api.seinet.com.mk/public/documents"
ATTACHMENT_URL = "https://api.seinet.com.mk/public/documents/attachment/"
HEADERS = {"Content-Type": "application/json"}

# Load pre-trained model for sentiment analysis from HuggingFace
sentiment_analysis_model = pipeline("sentiment-analysis")

# Функција за извлекување текст од PDF
def extract_pdf_text(pdf_content):
    text = ""
    try:
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if not text.strip():
            pdf_reader = PdfReader(BytesIO(pdf_content))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        # OCR if no text found
        if not text.strip():
            try:
                with fitz.open(stream=BytesIO(pdf_content), filetype="pdf") as pdf_document:
                    for page in pdf_document:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        text += ocr_text
            except Exception as ocr_error:
                print(f"OCR failed: {ocr_error}")
                text = "No text extracted from PDF (OCR failed)"
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
        text = "No text extracted from PDF (Error)"

    return text.strip()


# Функција за скипирање на документи
def fetch_documents():
    data_list = []
    page = 1

    while True:
        payload = {
            "issuerId": 0,
            "languageId": 1,  # За македонски јазик
            "channelId": 1,
            "dateFrom": "2024-11-01T00:00:00",
            "dateTo": "2024-12-23T23:59:59",
            "isPushRequest": False,
            "page": page
        }

        try:
            response = requests.post(BASE_URL, json=payload, headers=HEADERS)
            response.raise_for_status()
            json_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page {page}: {e}")
            break  # Прекини ако има грешка

        if not json_data.get('data'):
            print("No more data available.")
            break  # Прекини ако нема повеќе податоци

        for document in json_data.get('data', []):
            issuer_code = document.get('issuer', {}).get('code', 'N/A')
            content = document.get('content', 'N/A')
            description = document.get('layout', {}).get('description', 'N/A')
            published_date = document.get('publishedDate', '').split("T")[0]
            display_name = document.get('issuer', {}).get('localizedTerms', [{}])[0].get('displayName', 'N/A')
            content = html.unescape(content)
            content = re.sub(r'<[^>]*>', '', content)
            attachments = document.get('attachments', [])
            text_from_pdf = ""

            if attachments:
                for attachment in attachments:
                    attachment_id = attachment.get('attachmentId')
                    file_name = attachment.get('fileName', '')
                    if file_name and file_name.lower().endswith('.pdf'):
                        attachment_url = f"{ATTACHMENT_URL}{attachment_id}"
                        try:
                            attachment_response = requests.get(attachment_url)
                            attachment_response.raise_for_status()
                            pdf_content = attachment_response.content
                            text_from_pdf = extract_pdf_text(pdf_content)
                        except requests.exceptions.RequestException as e:
                            print(f"Failed to download PDF {file_name}: {e}")
            else:
                text_from_pdf = content if content.strip() else "No PDF text extracted"

            data_list.append({
                "Issuer Code": issuer_code,
                "Display Name": display_name,
                "Published Date": published_date,
                "Description": description,
                "Content": content,
                "Extracted PDF Text": text_from_pdf if text_from_pdf.strip() else "No PDF text extracted"
            })

        page += 1
        time.sleep(1)  # Паузирај за да не бидеш блокиран од API

    return pd.DataFrame(data_list)


# Превод и транслитерација
def transliterate_to_latin(text):
    return unidecode(text)


def translate_to_english(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='mk', dest='en')
        return translated.text
    except Exception as e:
        return str(e)


# Функција за анализа на сентимент со NLP модел
def label_sentiment_with_nlp(text):
    try:
        result = sentiment_analysis_model(text)
        sentiment = result[0]['label']
        confidence = result[0]['score']

        if sentiment == 'POSITIVE' and confidence > 0.5:
            return 'Positive'
        elif sentiment == 'NEGATIVE' and confidence > 0.5:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 'Neutral'


# Генерирање на препорака
def get_recommendation(row):
    if row['Positive'] > row['Negative']:
        return "Buy stocks"
    elif row['Negative'] > row['Positive']:
        return "Sell stocks"
    else:
        return "Hold"


def process_data():
    df = fetch_documents()

    # Применете транслитерација
    df['Display Name'] = df['Display Name'].apply(transliterate_to_latin)

    # Применете превод
    df['Description'] = df['Description'].apply(translate_to_english)
    df['Content'] = df['Content'].apply(lambda x: translate_to_english(x))

    # Анализа на сентимент со NLP модел
    df['sentiment_label'] = df['Content'].apply(label_sentiment_with_nlp)

    # Пресметување на процентите на сентименти за секоја компанија
    sentiment_counts = df.groupby('Display Name')['sentiment_label'].value_counts(normalize=True).unstack().fillna(
        0) * 100

    # Додаваме препораки
    sentiment_counts['Recommendation'] = sentiment_counts.apply(get_recommendation, axis=1)

    # Запишување на резултатите во CSV датотека
    sentiment_counts.to_csv('sentiment_counts.csv', index=True)

    print("Data successfully saved to sentiment_counts.csv")

    return sentiment_counts
