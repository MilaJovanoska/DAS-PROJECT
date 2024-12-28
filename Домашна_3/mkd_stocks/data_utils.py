#1. Подготовка на податоците
import pandas as pd
from datetime import datetime
from .models import Stock


def get_historical_data(issuer_name):
    # Извлечи податоци за избраниот издавач од базата
    data = Stock.objects.filter(issuer=issuer_name).values('date', 'last_price')

    # Претвори податоците во Pandas DataFrame
    df = pd.DataFrame(list(data))

    # Конверзија на 'date' од текст во datetime формат
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # Конверзија на 'last_price' од текст во нумерички формат
    df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')

    # Отстрани редови со невалидни (NaN) вредности
    df.dropna(inplace=True)

    # Сортирај ги податоците според датум
    df.sort_values(by='date', inplace=True)

    return df


#2.Нормализација на податоците

from sklearn.preprocessing import MinMaxScaler


def normalize_data(df):
    # Креирај MinMaxScaler за нормализација
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Примени нормализација на колоната 'last_price'
    df['normalized_price'] = scaler.fit_transform(df[['last_price']])

    return df, scaler


#3.Генерирање на временски прозорци

import numpy as np


def create_sequences(data, sequence_length):
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        # Земаме секвенца од податоци
        sequences.append(data[i:i + sequence_length])
        # Следната вредност како цел
        labels.append(data[i + sequence_length])

    return np.array(sequences), np.array(labels)

