import  tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense

import os

def train_lstm_model(X_train, y_train, X_test, y_test, model_path):
    # 1. Дефинирај LSTM модел
    model = tf.Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(1)  # Еден излез - цената
    ])

    # 2. Компилирај го моделот
    model.compile(optimizer='adam', loss='mse')

    # 3. Callback за рано запирање
    early_stopping = tf.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 4. Тренирај го моделот
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=20, batch_size=32, callbacks=[early_stopping])

    # 5. Создај папка за модели ако не постои
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Model path: {model_path}")
    # 6. Зачувување на моделот
    model.save(model_path)

    return model
