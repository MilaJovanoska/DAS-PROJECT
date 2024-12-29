import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from .models import Stock

# 1. Функција за добивање на историски податоци
# def get_historical_data(issuer_name):
#     """
#     Извлечи историски податоци за избраниот издавач од базата и претвори во Pandas DataFrame.
#     """
#     # Извлечи податоци од базата
#     data = Stock.objects.filter(issuer=issuer_name).values('date', 'last_price')
#
#     # Претвори податоците во DataFrame
#     df = pd.DataFrame(list(data))
#
#     # Конверзија на 'date' во datetime формат
#     df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
#
#     # Конверзија на 'last_price' во нумерички формат
#     df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')
#
#     # Отстрани редови со NaN вредности
#     df.dropna(inplace=True)
#
#     # Сортирај податоци според датум
#     df.sort_values(by='date', inplace=True)
#
#     return df

def get_historical_data(issuer_name):
    try:
        # Пребарување на податоците во базата
        # Осигури се дека имаш врска со базата и ги вадиш податоците
        stocks = Stock.objects.filter(Issuer=issuer_name)

        # Ако нема податоци, врати празен DataFrame
        if not stocks.exists():
            return pd.DataFrame()

        # Претвори податоците во DataFrame
        data = list(stocks.values('Date', 'Last Price'))
        df = pd.DataFrame(data)

        # Осигури се дека датумите се во правилен формат
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['Date'])

        # Претвори цените во бројки
        df['Last Price'] = pd.to_numeric(df['Last Price'].str.replace(',', '.', regex=False), errors='coerce')

        # Ако нема податоци, врати празен DataFrame
        return df if not df.empty else pd.DataFrame()

    except Exception as e:
        print(f"Error during data retrieval: {e}")
        return pd.DataFrame()  # Врати празен DataFrame во случај на грешка


# 2. Функција за нормализација на податоците
def normalize_data(df):
    """
    Нормализирај ја колоната 'last_price' со MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['normalized_price'] = scaler.fit_transform(df[['Last Price']])

    return df, scaler

def create_sequences(data, sequence_length):
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])  # се земаат последователни вредности
        labels.append(data[i + sequence_length])  # предвидување на следната вредност

    X = np.array(sequences)
    y = np.array(labels)
    return X, y


