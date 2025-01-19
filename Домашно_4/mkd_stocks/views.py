# Create your views here.
from django.contrib import messages
import tensorflow as tf
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Stock
import datetime
import json
from datetime import timedelta, datetime
import pandas as pd
from .data_utils import get_historical_data, normalize_data, create_sequences
from .train_lstm import train_lstm_model
from sklearn.model_selection import train_test_split

from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.http import JsonResponse


def home(request):
    return render(request, 'index.html')
def about_us(request):
    return render(request, 'about_us.html')
def sign_up(request):
    return render(request, 'sign_up.html')
def news(request):
    return render(request, 'stock_data.html')
def contact(request):
    return render(request, 'contact.html')

def technical_analysis(request):
    return render(request, 'technical_analysis.html')

def nlp_analysis(request):
    return render(request, 'nlp_analysis.html')

def lstm_analysis(request):
    return render(request, 'lstm_analysis.html')


def stock_list(request):
    query = request.GET.get('query', '').strip()
    date = request.GET.get('date', '').strip()
    table_data = []
    graph_data = []

    original_date = date

    if date:
        try:
            # Конверзија на датумот од формат YYYY-MM-DD во DD.MM.YYYY
            date = datetime.strptime(date, '%Y-%m-%d').strftime('%d.%m.%Y')
        except ValueError:
            messages.error(request, "Invalid date format. Please use a valid date.")
            date = ''

    if date and query:
        table_data = Stock.objects.filter(issuer__icontains=query, date=date).order_by('date')
        graph_data = table_data[:10]
    elif query:
        table_data = Stock.objects.filter(issuer__icontains=query).order_by('-date')[:15]
        all_graph_data = Stock.objects.filter(issuer__icontains=query).order_by('-date')
        graph_data = all_graph_data[:10]
    elif date:
        messages.error(request, "Please provide an issuer along with the date.")
    else:
        table_data = Stock.objects.all().order_by('-date')[:15]

    return render(request, 'stock_data.html', {
        'stocks': table_data,
        'graph_data': graph_data,
        'query': query,
        'date': original_date
    })

from django.http import JsonResponse
from datetime import datetime
from .models import Stock
import pandas as pd


def get_stock_data(request):
    issuer = request.GET.get('issuer', '').strip()
    time_period = request.GET.get('time_period', '').strip()

    if not issuer:
        return JsonResponse({"error": "Issuer is required."})

    try:
        # 🗂️ Извлекување на податоците од базата
        stocks = Stock.objects.filter(issuer=issuer).values('date', 'last_price')

        if not stocks.exists():
            return JsonResponse({"error": f"No data found for issuer {issuer}."})

        # 📊 Претворање на податоците во DataFrame
        df = pd.DataFrame(list(stocks))

        # ✅ Валидација на колоните
        if 'date' not in df.columns or 'last_price' not in df.columns:
            return JsonResponse({"error": "Required columns ('date', 'last_price') are missing in the data."})

        # 🗓️ Конверзија на датумите
        df['date'] = pd.to_datetime(df['date'], format="%d.%m.%Y", errors='coerce')
        df = df.dropna(subset=['date'])

        # 💵 Конверзија на цените во бројки
        df['last_price'] = pd.to_numeric(
            df['last_price'].astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False),
            errors='coerce'
        )
        df = df.dropna(subset=['last_price'])

        # 📅 Сортирање на податоците според датум
        df = df.sort_values(by='date')

        # ➗ Извлекување на секој 3-ти податок
        df = df.iloc[::3]

        # 📤 Подготвување на податоците за одговор
        dates = df['date'].dt.strftime("%Y-%m-%d").tolist()
        prices = df['last_price'].tolist()

        return JsonResponse({
            "dates": dates,
            "prices": prices,
        })

    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"})



def get_indicators(request):
    issuer = request.GET.get('issuer', '').strip()
    time_period = request.GET.get('time_period', '').strip()

    if not issuer or not time_period:
        return JsonResponse({'error': 'Missing parameters'}, status=400)

    time_period_days = {
        "1-day": 1,
        "1-week": 7,
        "1-month": 30
    }.get(time_period, 1)

    stocks = Stock.objects.filter(issuer__iexact=issuer)

    if not stocks.exists():
        return JsonResponse({'error': 'No data found for the given company'}, status=404)

    data = []
    for stock in stocks:
        try:
            parts = stock.date.strip().split('.')
            day = parts[0].zfill(2)
            month = parts[1].zfill(2)
            year = parts[2]
            standardized_date = f"{day}.{month}.{year}"
            date = datetime.strptime(standardized_date, '%d.%m.%Y')
            price = float(stock.last_price.replace('.', '').replace(',', '.').strip())
            data.append({'date': date, 'price': price})

        except (ValueError, IndexError) as e:
            print(f"Skipping invalid data: {e}")
            continue

    if not data:
        return JsonResponse({'error': 'No valid data to process'}, status=400)

    df = pd.DataFrame(data)
    df.sort_values('date', inplace=True)

    end_date = df['date'].max()
    start_date = end_date - timedelta(days=time_period_days)
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Пополнување на податоци ако има помалку од 14 записи
    if len(df) < 14:
        last_record = df.iloc[-1]  # Последен запис
        missing_records = 14 - len(df)
        new_rows = [last_record.to_dict()] * missing_records  # Додај дупликати
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.sort_values('date', inplace=True)

    if len(df) < 14:
        return JsonResponse({'error': 'Not enough data for calculations'}, status=400)

    try:
        df['RSI'] = calculate_rsi(df['price']).round(2).fillna("N/A")
        df['SO'] = calculate_stochastic_oscillator(df['price']).round(2).fillna("N/A")
        df['MACD'] = calculate_macd(df['price']).round(2).fillna("N/A")
        df['CCI'] = calculate_cci(df['price']).round(2).fillna("N/A")
        df['ATR'] = calculate_atr(df['price']).round(2).fillna("N/A")
        df['SMA'] = df['price'].rolling(window=5).mean().round(2).fillna("N/A")
        df['EMA'] = df['price'].ewm(span=5, adjust=False).mean().round(2).fillna("N/A")
        df['WMA'] = df['price'].rolling(window=5).apply(
            lambda prices: pd.Series(prices).dot(range(1, 6)) / 15, raw=True
        ).round(2).fillna("N/A")
        df['HMA'] = calculate_hma(df['price']).round(2).fillna("N/A")
        df['AMA'] = calculate_ama(df['price']).round(2).fillna("N/A")
    except Exception as e:
        print(f"Error during calculations: {e}")
        return JsonResponse({'error': 'Error during calculations'}, status=500)

    indicators = {
        "oscillators": {
            "RSI": df['RSI'].iloc[-1],
            "SO": df['SO'].iloc[-1],
            "MACD": df['MACD'].iloc[-1],
            "CCI": df['CCI'].iloc[-1],
            "ATR": df['ATR'].iloc[-1],
        },
        "moving_averages": {
            "SMA": df['SMA'].iloc[-1],
            "EMA": df['EMA'].iloc[-1],
            "WMA": df['WMA'].iloc[-1],
            "HMA": df['HMA'].iloc[-1],
            "AMA": df['AMA'].iloc[-1],
        }
    }

    return JsonResponse(indicators)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_oscillator(prices, period=14):
    low = prices.rolling(window=period).min()
    high = prices.rolling(window=period).max()
    so = ((prices - low) / (high - low)) * 100
    return so

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal

def calculate_cci(prices, period=20):
    mean_price = prices.rolling(window=period).mean()
    mean_deviation = prices.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (prices - mean_price) / (0.015 * mean_deviation)
    return cci

def calculate_atr(prices, period=14):
    return prices.rolling(window=period).std()

def calculate_hma(prices, period=14):
    half_length = int(period / 2)
    sqrt_length = int(period ** 0.5)
    weighted_ma = prices.rolling(window=half_length).mean()
    hma = 2 * weighted_ma - prices.rolling(window=period).mean()
    return hma.rolling(window=sqrt_length).mean()

def calculate_ama(prices, period=10):
    return prices.ewm(span=period).mean()


def login_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        # Authenticate the user
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to the home page (or wherever you want)
        else:
            return render(request, 'log_in.html', {'error': 'Invalid email or password'})

    return render(request, 'log_in.html')


import pandas as pd
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Функција за читање на CSV податоци
def read_sentiment_data():
    file_path = './mkd_stocks/sentiment_counts.csv'  # Датотеката е во истата папка како views.py
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found. Check the file path.")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

@csrf_exempt
def company_view(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

    print("company_view called")

    # Parse JSON request
    data, error = parse_json_request(request)
    if error:
        return JsonResponse({'status': 'error', 'message': error}, status=400)

    company_name = data.get('company_name')
    if not company_name:
        return JsonResponse({'status': 'error', 'message': 'Company name is required'}, status=400)

    # Read sentiment data from CSV
    df = read_sentiment_data()
    print(f"DataFrame loaded. Columns: {df.columns}")

    # Search for company data
    company_data = df[df['Issuer Code'].str.strip() == company_name.strip().upper()]
    if company_data.empty:
        print("No data found for company:", company_name)
        return JsonResponse({'status': 'error', 'message': 'No data available for this company'}, status=404)

    # Get sentiment and recommendation
    company_row = company_data.iloc[0]
    sentiment_label, recommendation, pie_data, bar_data = get_company_sentiment(company_row)

    # Return the response
    return JsonResponse({
        'status': 'success',
        'recommendation': recommendation,
        'sentiment_label': sentiment_label,
        'pie_data': pie_data,
        'bar_data': bar_data
    })




from django.http import JsonResponse
from .data_utils import get_historical_data, normalize_data, create_sequences
from .train_lstm import train_lstm_model
from sklearn.model_selection import train_test_split

import numpy as np
import os


def train_model_view(request, issuer_name):
    try:
        # 1. Подготви податоци
        print(f"Training LSTM model for issuer: {issuer_name}")
        df = get_historical_data(issuer_name)

        if df.empty:
            return JsonResponse({"error": "No data found for the specified issuer."}, status=400)

        df, scaler = normalize_data(df)
        sequence_length = 60
        data = df['normalized_price'].values

        if len(data) < sequence_length:
            return JsonResponse({"error": "Not enough data to create sequences for training."}, status=400)

        X, y = create_sequences(data, sequence_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # 2. Тренирај го LSTM моделот
        model = train_lstm_model(X_train, y_train, X_test, y_test)

        # 3. Создај папка за модели ако не постои
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{issuer_name}_lstm_model.h5"
        model.save(model_path)  # Се снима со име на издавачот

        # 4. Врати одговор
        return JsonResponse({
            "message": "Model trained and saved successfully!",
            "model_path": model_path,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape
        })

    except Exception as e:
        return JsonResponse({"error": f"An error occurred during training: {str(e)}"}, status=500)


def predict_stock_prices(request, issuer_name):
    try:
        # 1. Провери дали моделот постои
        model_path = f"models/{issuer_name}_lstm_model.h5"
        if not os.path.exists(model_path):
            return JsonResponse({"error": "Model not found for the specified issuer."}, status=404)

        # 2. Вчитај го моделот
        #model = tf.keras.load_model(model_path)
        model = tf.keras.models.load_model(model_path)

        # 3. Подготви податоци
        df = get_historical_data(issuer_name)
        if df.empty:
            return JsonResponse({"error": "No data found for the specified issuer."}, status=400)

        df, scaler = normalize_data(df)
        sequence_length = 60
        data = df['normalized_price'].values
        if len(data) < sequence_length:
            return JsonResponse({"error": "Not enough data to make predictions."}, status=400)

        X, _ = create_sequences(data, sequence_length)

        # 4. Направи предвидувања
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)  # Врати ги во оригиналниот опсег

        # 5. Форматирај го резултатот
        predicted_prices = predictions.flatten().tolist()
        return JsonResponse({"predicted_prices": predicted_prices})

    except Exception as e:
        return JsonResponse({"error": f"An error occurred during prediction: {str(e)}"}, status=500)

