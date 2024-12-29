import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from .models import Stock


# 1. Функција за добивање на историски податоци
def get_historical_data(issuer_name):
    """
    Извлечи историски податоци за избраниот издавач од базата и претвори во Pandas DataFrame.
    """
    # Извлечи податоци од базата
    data = Stock.objects.filter(issuer=issuer_name).values('date', 'last_price')

    # Претвори податоците во DataFrame
    df = pd.DataFrame(list(data))

    # Конверзија на 'date' во datetime формат
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # Конверзија на 'last_price' во нумерички формат
    df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')

    # Отстрани редови со NaN вредности
    df.dropna(inplace=True)

    # Сортирај податоци според датум
    df.sort_values(by='date', inplace=True)

    return df


# 2. Функција за нормализација на податоците
def normalize_data(df):
    """
    Нормализирај ја колоната 'last_price' со MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['normalized_price'] = scaler.fit_transform(df[['last_price']])

    return df, scaler


# 3. Функција за генерирање на временски прозорци
def create_sequences(data, sequence_length):
    """
    Генерира временски прозорци од податоците за LSTM моделот.
    """
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        # Земаме секвенца со големина sequence_length
        sequences.append(data[i:i + sequence_length])
        # Следната вредност како цел
        labels.append(data[i + sequence_length])

    return np.array(sequences), np.array(labels)

