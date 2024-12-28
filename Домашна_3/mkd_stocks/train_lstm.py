import tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense


def train_lstm_model(X_train, y_train, X_test, y_test):
    # 1. Дефинирај LSTM модел
    model = tf.keras.Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(1)  # Еден излез - цената
    ])

    # 2. Компилирај го моделот
    model.compile(optimizer='adam', loss='mse')

    # 3. Тренирај го моделот
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    return model
