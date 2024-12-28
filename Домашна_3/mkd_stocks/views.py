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


def get_stock_data(request):
    issuer = request.GET.get('issuer', '').strip()
    time_period = request.GET.get('time_period', '').strip()

    if not issuer:
        return JsonResponse({"error": "Issuer is required."})

    try:
        # Извлекување на податоците од базата
        stocks = Stock.objects.filter(issuer=issuer)

        if not stocks.exists():
            return JsonResponse({"error": f"No data found for issuer {issuer}."})

        # Претворање на податоците во Python објекти за сортирање
        stock_data = []
        for stock in stocks:
            try:
                # Конверзија на датумот
                date = datetime.strptime(stock.date, "%d.%m.%Y")
                # Конверзија на цената од македонски формат
                last_price = float(stock.last_price.replace('.', '').replace(',', '.'))
                stock_data.append((date, last_price))
            except ValueError:
                continue

        # Сортирање на податоците според датумот
        stock_data.sort(key=lambda x: x[0])

        # Извлекување на секој 3-ти податок
        dates = []
        prices = []
        for i, (date, price) in enumerate(stock_data):
            if i % 3 == 0:
                dates.append(date.strftime("%Y-%m-%d"))
                prices.append(price)

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
    return render(request, 'log_in.html')
# def read_sentiment_data():
#     file_path = 'sentiment_counts.csv'  # Постави точна патека до твојот CSV фајл
#     try:
#         df = pd.read_csv(file_path)
#         print(df.head())  # Испечати ги првите неколку редови за проверка
#         return df
#     except FileNotFoundError:
#         raise FileNotFoundError("CSV file not found. Check the file path.")
#     except pd.errors.EmptyDataError:
#         raise ValueError("CSV file is empty.")
#     except Exception as e:
#         raise Exception(f"Error reading CSV: {e}")
#
#
# # Главен view за компанија и препораки
# @csrf_exempt
# def company_view(request):
#     # Иницијализација на променливи
#     recommendation = ""
#     sentiment_label = ""
#     pie_data = [0, 0, 0]  # Дефинирање на Pie chart податоци
#     bar_data = [0, 0, 0]  # Дефинирање на Bar chart податоци
#
#     if request.method == 'POST':
#         try:
#             # Читање на податоци од AJAX барањето
#             data = json.loads(request.body)
#             company_name = data.get('company_name')
#
#             if not company_name:
#                 return JsonResponse({'error': 'Company name is required'}, status=400)
#
#             # Читање на податоци од CSV
#             df = read_sentiment_data()
#
#             # Филтрирање на податоци за избраната компанија
#             company_data = df[df['Company'] == company_name]
#
#             if not company_data.empty:
#                 company_row = company_data.iloc[0]  # Земете го првиот резултат
#
#                 # Определување на сентиментот според проценти
#                 positive = company_row.get('Positive', 0)
#                 negative = company_row.get('Negative', 0)
#
#                 if positive > 50:
#                     sentiment_label = "Positive"
#                 elif negative > 50:
#                     sentiment_label = "Negative"
#                 else:
#                     sentiment_label = "Neutral"
#
#                 # Додавање препорака
#                 recommendation = company_row.get('Recommendation', 'No recommendation')
#
#                 print(f"Recommendation: {recommendation}")
#                 print(f"Sentiment: {sentiment_label}")
#
#                 # Генерирање податоци за графикони
#                 if sentiment_label == "Positive":
#                     pie_data = [70, 20, 10]
#                     bar_data = [15, 10, 5]
#                 elif sentiment_label == "Negative":
#                     pie_data = [20, 70, 10]
#                     bar_data = [10, 15, 5]
#                 else:
#                     pie_data = [30, 30, 40]
#                     bar_data = [12, 8, 10]
#             else:
#                 recommendation = "No data available for this company"
#                 pie_data = [0, 0, 100]
#                 bar_data = [0, 0, 0]
#
#             # Врати JSON одговор
#             return JsonResponse({
#                 'recommendation': recommendation,
#                 'sentiment_label': sentiment_label,
#                 'pie_data': pie_data,
#                 'bar_data': bar_data
#             })
#
#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON format'}, status=400)
#         except FileNotFoundError as e:
#             return JsonResponse({'error': str(e)}, status=500)
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#
#     return JsonResponse({'error': 'Invalid request method'}, status=405)
import os
import pandas as pd
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

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
    if request.method == 'POST':
        try:
            print("company_view called")  # Потврди дека функцијата е повикана

            # Читање на податоци од JSON барањето
            data = json.loads(request.body)
            print("Received data:", data)  # Испечати ги податоците за проверка

            company_name = data.get('company_name')
            if not company_name:
                return JsonResponse({'status': 'error', 'message': 'Company name is required'}, status=400)

            # Читање на податоците од CSV
            print("Calling read_sentiment_data()...")
            df = read_sentiment_data()
            print("DataFrame loaded. Columns:", df.columns)  # Испечати ги сите колони во CSV-то

            # Пребарување на компанија според `Issuer Code`
            print(f"Searching for company: {company_name.strip().upper()}")
            company_data = df[df['Issuer Code'].str.strip() == company_name.strip().upper()]
            print("Filtered company data:", company_data)

            if not company_data.empty:
                # Превземање на првиот ред од резултатите
                company_row = company_data.iloc[0]
                print("Company row:", company_row)

                positive = company_row.get('Positive', 0)
                negative = company_row.get('Negative', 0)
                neutral = company_row.get('Neutral', 0)

                # Одредување на сентиментот според процентите
                if positive > 50:
                    sentiment_label = "Positive"
                elif negative > 50:
                    sentiment_label = "Negative"
                else:
                    sentiment_label = "Neutral"

                print(f"Sentiment: {sentiment_label}")

                # Препорака
                recommendation = company_row.get('Recommendation', 'No recommendation')
                print(f"Recommendation: {recommendation}")

                # Податоци за графикони
                pie_data = [positive, negative, neutral]
                bar_data = [positive, negative, neutral]
                print(f"Pie Data: {pie_data}")
                print(f"Bar Data: {bar_data}")

                return JsonResponse({
                    'status': 'success',
                    'recommendation': recommendation,
                    'sentiment_label': sentiment_label,
                    'pie_data': pie_data,
                    'bar_data': bar_data
                })
            else:
                print("No data found for company:", company_name)
                return JsonResponse({
                    'status': 'error',
                    'message': 'No data available for this company'
                }, status=404)

        except json.JSONDecodeError:
            print("Invalid JSON format")
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)
        except Exception as e:
            print(f"Unexpected error: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    print("Invalid request method")
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)



#Tuka e tarin
def train_model_view(request, issuer_name):
    # 1. Подготви податоци
    df = get_historical_data(issuer_name)
    df, scaler = normalize_data(df)
    sequence_length = 60
    data = df['normalized_price'].values
    X, y = create_sequences(data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # 2. Тренирај го LSTM моделот
    model = train_lstm_model(X_train, y_train, X_test, y_test)

    # 3. Сними го тренираниот модел
    model.save('lstm_model.h5')  # Се снима во истата папка каде што е проектот

    # 4. Врати одговор
    return JsonResponse({
        "message": "Model trained and saved successfully!",
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape
    })


from django.http import JsonResponse
import numpy as np
from .data_utils import get_historical_data, normalize_data


def predict_price_view(request, issuer_name):
    # 1. Извлечи и нормализирај податоци
    df = get_historical_data(issuer_name)
    df, scaler = normalize_data(df)
    sequence_length = 60
    data = df['normalized_price'].values[-sequence_length:]  # Земи ги последните 60 записи

    # 2. Вчитај го зачуваниот модел
    model = tf.keras.load_model('lstm_model.h5')

    # 3. Подготви податоци за предвидување
    input_data = np.expand_dims(data, axis=0)  # Претвори во формат што моделот го очекува

    # 4. Направи предвидување
    predicted_price = model.predict(input_data)
    predicted_price = scaler.inverse_transform([[predicted_price[0][0]]])  # Де-нормализација

    # 5. Врати го резултатот
    return JsonResponse({
        "predicted_price": predicted_price[0][0]
    })


