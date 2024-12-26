# Create your views here.
from django.http import JsonResponse
from .models import Stock
from django.contrib import messages
from .models import Stock
from datetime import datetime
from django.shortcuts import render
import pandas as pd

import datetime
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

from django.shortcuts import render

# def stock_list(request):
#     query = request.GET.get('query', '').strip()
#     date = request.GET.get('date', '').strip()
#     table_data = []
#     graph_data = []
#
#     original_date = date
#
#
#     if date:
#         try:
#
#             date = datetime.strptime(date, '%Y-%m-%d').strftime('%d.%m.%Y')
#
#             day, month, year = date.split('.')
#             date = f"{day}.{int(month)}.{year}"
#         except ValueError:
#             messages.error(request, "Invalid date format. Please use a valid date.")
#             date = ''
#
#
#     if date and query:
#
#         table_data = Stock.objects.filter(issuer__icontains=query, date=date).order_by('date')
#         graph_data = table_data[:10]
#     elif query:
#
#         table_data = Stock.objects.filter(issuer__icontains=query).order_by('-date')[:15]
#         all_graph_data = Stock.objects.filter(issuer__icontains=query).order_by('-date')
#         graph_data = all_graph_data[:10]
#     elif date:
#
#         messages.error(request, "Please provide an issuer along with the date.")
#     else:
#
#         table_data = Stock.objects.all().order_by('-date')[:15]
#
#
#     return render(request, 'stock_data.html', {
#         'stocks': table_data,
#         'graph_data': graph_data,
#         'query': query,
#         'date': original_date
#     })
#
#
#
#
#
# def get_stock_data(request):
#     issuer = request.GET.get('issuer', '').strip()
#     time_period = request.GET.get('time_period', '').strip()
#
#     if not issuer:
#         return JsonResponse({"error": "Issuer is required."})
#
#     try:
#         # Извлекување на податоците од базата
#         stocks = Stock.objects.filter(issuer=issuer)
#
#         if not stocks.exists():
#             return JsonResponse({"error": f"No data found for issuer {issuer}."})
#
#         # Претворање на податоците во Python објекти за сортирање
#         stock_data = []
#         for stock in stocks:
#             try:
#                 date = datetime.datetime.strptime(stock.date, "%d.%m.%Y")  # Претворање на датум
#                 last_price = float(stock.last_price.replace('.', '').replace(',', '.'))  # Претворање на цена
#                 stock_data.append((date, last_price))
#             except ValueError:
#                 continue  # Прескокнување ако датумот не е валиден
#
#         # Сортирање на податоците според датумот
#         stock_data.sort(key=lambda x: x[0])
#
#         # Извлекување на секој 3-ти податок
#         dates = []
#         prices = []
#         for i, (date, price) in enumerate(stock_data):
#             if i % 3 == 0:  # Земаме секој 3-ти податок
#                 dates.append(date.strftime("%Y-%m-%d"))  # Форматиран датум
#                 prices.append(price)
#
#         return JsonResponse({
#             "dates": dates,
#             "prices": prices,
#         })
#
#     except Exception as e:
#         return JsonResponse({"error": f"An error occurred: {str(e)}"})

from django.http import JsonResponse
from django.shortcuts import render
from .models import Stock
from datetime import datetime, timedelta
import pandas as pd

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

import  numpy as np
# def get_indicators(request):
#     from datetime import timedelta  # Осигурај се дека го повикуваме точно
#     issuer = request.GET.get('issuer', '').strip()
#     time_period = request.GET.get('time_period', '').strip()
#
#     if not issuer or not time_period:
#         return JsonResponse({'error': 'Missing parameters'}, status=400)
#
#     # Претвори го времетраењето во денови
#     time_period_days = {
#         "1-day": 1,
#         "1-week": 7,
#         "1-month": 30
#     }.get(time_period, 1)
#
#     # Преземи податоци од базата
#     stocks = Stock.objects.filter(issuer__iexact=issuer)
#
#     if not stocks.exists():
#         return JsonResponse({'error': 'No data found for the given company'}, status=404)
#
#     # Претвори податоци во pandas DataFrame
#     data = []
#     for stock in stocks:
#         try:
#             date = datetime.strptime(stock.date.strip(), '%d.%m.%Y')  # Претворање датум
#             price = float(stock.last_price.replace('.', '').replace(',', '.').strip())  # Претворање цена
#             data.append({'date': date, 'price': price})
#         except ValueError:
#             continue
#
#     if not data:
#         return JsonResponse({'error': 'No valid data to process'}, status=400)
#
#     df = pd.DataFrame(data)
#     df.sort_values('date', inplace=True)  # Сортирање по датум
#
#     # Филтрирај според временски период
#     end_date = df['date'].max()
#     start_date = end_date - timedelta(days=time_period_days)
#     df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
#
#     if df.empty or len(df) < 14:  # Ако има премалку податоци за пресметки
#         return JsonResponse({'error': 'Not enough data for calculations'}, status=404)
#
#     # Пресметај осцилатори и moving averages со default вредности за недостасувачки податоци
#     try:
#         df['RSI'] = calculate_rsi(df['price']).fillna("N/A")
#         df['SO'] = calculate_stochastic_oscillator(df['price']).fillna("N/A")
#         df['MACD'] = calculate_macd(df['price']).fillna("N/A")
#         df['CCI'] = calculate_cci(df['price']).fillna("N/A")
#         df['ATR'] = calculate_atr(df['price']).fillna("N/A")
#     except Exception as e:
#         print(f"Error calculating oscillators: {e}")
#
#     # Пресметај Moving Averages
#     try:
#         df['SMA'] = df['price'].rolling(window=5).mean().fillna("N/A")
#         df['EMA'] = df['price'].ewm(span=5, adjust=False).mean().fillna("N/A")
#         df['WMA'] = df['price'].rolling(window=5).apply(
#             lambda prices: pd.Series(prices).dot(range(1, 6)) / 15, raw=True
#         ).fillna("N/A")
#         df['HMA'] = calculate_hma(df['price']).fillna("N/A")
#         df['AMA'] = calculate_ama(df['price']).fillna("N/A")
#     except Exception as e:
#         print(f"Error calculating moving averages: {e}")
#
#     # Земи последните вредности за секоја метрика
#     indicators = {
#         "oscillators": {
#             "RSI": df['RSI'].iloc[-1] if len(df['RSI']) > 0 else "N/A",
#             "SO": df['SO'].iloc[-1] if len(df['SO']) > 0 else "N/A",
#             "MACD": df['MACD'].iloc[-1] if len(df['MACD']) > 0 else "N/A",
#             "CCI": df['CCI'].iloc[-1] if len(df['CCI']) > 0 else "N/A",
#             "ATR": df['ATR'].iloc[-1] if len(df['ATR']) > 0 else "N/A",
#         },
#         "moving_averages": {
#             "SMA": df['SMA'].iloc[-1] if len(df['SMA']) > 0 else "N/A",
#             "EMA": df['EMA'].iloc[-1] if len(df['EMA']) > 0 else "N/A",
#             "WMA": df['WMA'].iloc[-1] if len(df['WMA']) > 0 else "N/A",
#             "HMA": df['HMA'].iloc[-1] if len(df['HMA']) > 0 else "N/A",
#             "AMA": df['AMA'].iloc[-1] if len(df['AMA']) > 0 else "N/A",
#         }
#     }
#
#     return JsonResponse(indicators)
# def get_indicators(request):
#     from datetime import timedelta
#
#     issuer = request.GET.get('issuer', '').strip()
#     time_period = request.GET.get('time_period', '').strip()
#
#     if not issuer or not time_period:
#         return JsonResponse({'error': 'Missing parameters'}, status=400)
#
#     # Претвори го времетраењето во денови
#     time_period_days = {
#         "1-day": 1,
#         "1-week": 7,
#         "1-month": 30
#     }.get(time_period, 1)
#
#     # Преземи податоци од базата
#     stocks = Stock.objects.filter(issuer__iexact=issuer)
#
#     if not stocks.exists():
#         return JsonResponse({'error': 'No data found for the given company'}, status=404)
#
#     # Претвори податоци во pandas DataFrame
#     data = []
#     for stock in stocks:
#         try:
#             date = datetime.strptime(stock.date.strip(), '%d.%m.%Y')
#             price = float(stock.last_price.replace('.', '').replace(',', '.').strip())
#             data.append({'date': date, 'price': price})
#         except ValueError as e:
#             print(f"Skipping invalid data: {e}")
#             continue
#
#     if not data:
#         return JsonResponse({'error': 'No valid data to process'}, status=400)
#
#     df = pd.DataFrame(data)
#     df.sort_values('date', inplace=True)
#
#     # Филтрирај според временски период
#     end_date = df['date'].max()
#     start_date = end_date - timedelta(days=time_period_days)
#     df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
#
#     # Додади податоци ако фалат за пресметки
#     if len(df) < 14:
#         print("Not enough data for complete calculation. Using fallback.")
#         df = df.tail(14)  # Земаме последни 14 записи за да ги изведеме пресметките
#
#     if df.empty:
#         return JsonResponse({'error': 'No data available for the selected time period'}, status=404)
#
#     try:
#         # Пресметај осцилатори
#         df['RSI'] = calculate_rsi(df['price']).fillna("N/A")
#         df['SO'] = calculate_stochastic_oscillator(df['price']).fillna("N/A")
#         df['MACD'] = calculate_macd(df['price']).fillna("N/A")
#         df['CCI'] = calculate_cci(df['price']).fillna("N/A")
#         df['ATR'] = calculate_atr(df['price']).fillna("N/A")
#
#         # Пресметај Moving Averages
#         df['SMA'] = df['price'].rolling(window=5).mean().fillna("N/A")
#         df['EMA'] = df['price'].ewm(span=5, adjust=False).mean().fillna("N/A")
#         df['WMA'] = df['price'].rolling(window=5).apply(
#             lambda prices: pd.Series(prices).dot(range(1, 6)) / 15, raw=True
#         ).fillna("N/A")
#         df['HMA'] = calculate_hma(df['price']).fillna("N/A")
#         df['AMA'] = calculate_ama(df['price']).fillna("N/A")
#
#     except Exception as e:
#         print(f"Error during calculations: {e}")
#         return JsonResponse({'error': 'Error during calculations'}, status=500)
#
#     # Земи последните вредности за секоја метрика
#     indicators = {
#         "oscillators": {
#             "RSI": df['RSI'].iloc[-1] if len(df['RSI']) > 0 else "N/A",
#             "SO": df['SO'].iloc[-1] if len(df['SO']) > 0 else "N/A",
#             "MACD": df['MACD'].iloc[-1] if len(df['MACD']) > 0 else "N/A",
#             "CCI": df['CCI'].iloc[-1] if len(df['CCI']) > 0 else "N/A",
#             "ATR": df['ATR'].iloc[-1] if len(df['ATR']) > 0 else "N/A",
#         },
#         "moving_averages": {
#             "SMA": df['SMA'].iloc[-1] if len(df['SMA']) > 0 else "N/A",
#             "EMA": df['EMA'].iloc[-1] if len(df['EMA']) > 0 else "N/A",
#             "WMA": df['WMA'].iloc[-1] if len(df['WMA']) > 0 else "N/A",
#             "HMA": df['HMA'].iloc[-1] if len(df['HMA']) > 0 else "N/A",
#             "AMA": df['AMA'].iloc[-1] if len(df['AMA']) > 0 else "N/A",
#         }
#     }
#
#     return JsonResponse(indicators)

# Помошни функции за пресметка на осцилатори и движечки просеци
def get_indicators(request):
    from datetime import timedelta

    issuer = request.GET.get('issuer', '').strip()
    time_period = request.GET.get('time_period', '').strip()

    if not issuer or not time_period:
        return JsonResponse({'error': 'Missing parameters'}, status=400)

    # Претвори го времетраењето во денови
    time_period_days = {
        "1-day": 1,
        "1-week": 7,
        "1-month": 30
    }.get(time_period, 1)

    # Преземи податоци од базата
    stocks = Stock.objects.filter(issuer__iexact=issuer)

    if not stocks.exists():
        return JsonResponse({'error': 'No data found for the given company'}, status=404)

    # Претвори податоци во pandas DataFrame
    data = []
    for stock in stocks:
        try:
            date = datetime.strptime(stock.date.strip(), '%d.%m.%Y')
            price = float(stock.last_price.replace('.', '').replace(',', '.').strip())
            data.append({'date': date, 'price': price})
        except ValueError as e:
            print(f"Skipping invalid data: {e}")
            continue

    if not data:
        return JsonResponse({'error': 'No valid data to process'}, status=400)

    df = pd.DataFrame(data)
    df.sort_values('date', inplace=True)

    # Филтрирај според временски период
    end_date = df['date'].max()
    start_date = end_date - timedelta(days=time_period_days)
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Осигури се дека има минимум 14 записи за валидни пресметки
    if len(df) < 14:
        print("Not enough data for full calculations. Adding more records.")
        df = df.tail(14)  # Земаме последни 14 записи

    if df.empty:
        return JsonResponse({'error': 'No data available for the selected time period'}, status=404)

    try:
        # Пресметај осцилатори
        df['RSI'] = calculate_rsi(df['price']).fillna("N/A")
        df['SO'] = calculate_stochastic_oscillator(df['price']).fillna("N/A")
        df['MACD'] = calculate_macd(df['price']).fillna("N/A")
        df['CCI'] = calculate_cci(df['price']).fillna("N/A")
        df['ATR'] = calculate_atr(df['price']).fillna("N/A")

        # Пресметај Moving Averages
        df['SMA'] = df['price'].rolling(window=5).mean().fillna("N/A")
        df['EMA'] = df['price'].ewm(span=5, adjust=False).mean().fillna("N/A")
        df['WMA'] = df['price'].rolling(window=5).apply(
            lambda prices: pd.Series(prices).dot(range(1, 6)) / 15, raw=True
        ).fillna("N/A")
        df['HMA'] = calculate_hma(df['price']).fillna("N/A")
        df['AMA'] = calculate_ama(df['price']).fillna("N/A")

    except Exception as e:
        print(f"Error during calculations: {e}")
        return JsonResponse({'error': 'Error during calculations'}, status=500)

    # Земи последните вредности за секоја метрика
    indicators = {
        "oscillators": {
            "RSI": df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else "N/A",
            "SO": df['SO'].iloc[-1] if not pd.isna(df['SO'].iloc[-1]) else "N/A",
            "MACD": df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else "N/A",
            "CCI": df['CCI'].iloc[-1] if not pd.isna(df['CCI'].iloc[-1]) else "N/A",
            "ATR": df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else "N/A",
        },
        "moving_averages": {
            "SMA": df['SMA'].iloc[-1] if not pd.isna(df['SMA'].iloc[-1]) else "N/A",
            "EMA": df['EMA'].iloc[-1] if not pd.isna(df['EMA'].iloc[-1]) else "N/A",
            "WMA": df['WMA'].iloc[-1] if not pd.isna(df['WMA'].iloc[-1]) else "N/A",
            "HMA": df['HMA'].iloc[-1] if not pd.isna(df['HMA'].iloc[-1]) else "N/A",
            "AMA": df['AMA'].iloc[-1] if not pd.isna(df['AMA'].iloc[-1]) else "N/A",
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
