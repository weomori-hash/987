import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Загрузка и предобработка данных (набор MNIST)
# Набор содержит 60 000 обучающих и 10 000 тестовых изображений цифр от 0 до 9
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем значения пикселей (переводим из диапазона 0-255 в 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Создание архитектуры нейросети
model = models.Sequential([
    # Преобразуем матрицу 28x28 пикселей в плоский вектор из 784 элементов
    layers.Flatten(input_shape=(28, 28)), 
    
    # Скрытый полносвязный слой: 128 нейронов, функция активации ReLU
    layers.Dense(128, activation='relu'), 
    
    # Слой Dropout для предотвращения переобучения (случайно отключает 20% нейронов)
    layers.Dropout(0.2), 
    
    # Выходной слой: 10 нейронов (по одному на каждую цифру от 0 до 9)
    # Функция Softmax превращает выходы в вероятности
    layers.Dense(10, activation='softmax')
])

# 3. Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Обучение модели
print("--- Запуск обучения ---")
model.fit(x_train, y_train, epochs=5)

# 5. Проверка точности на тестовых данных
print("\n--- Оценка точности на тестовых данных ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nТочность модели на тесте: {test_acc * 100:.2f}%")

# 6. Сохранение готовой модели для дальнейшего использования
model.save('mnist_model.h5')
print("Модель сохранена как 'mnist_model.h5'")
