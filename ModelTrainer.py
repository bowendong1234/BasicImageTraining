# import tensorflow as tf
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import matplotlib.pyplot as plt

# # Data Preprocessing
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# train_generator = datagen.flow_from_directory(
#     'data',
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )

# validation_generator = datagen.flow_from_directory(
#     'data',
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )

# # Model Definition
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Model Training
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=validation_generator
# )

# # Model Evaluation
# loss, accuracy = model.evaluate(validation_generator)
# print(f'Validation accuracy: {accuracy*100:.2f}%')

# # Plotting training history
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

# # Save the model
# model.save('model.h5')
