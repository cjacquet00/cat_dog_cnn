# cat_dog_cnn

# Convolutional Neural Network (CNN) for Image Classification

This repository contains a Python script implementing a Convolutional Neural Network (CNN) for image classification using the Keras library. The CNN is designed to classify images into binary categories and includes data augmentation techniques to enhance model robustness.

## Data Preprocessing

The script uses the `ImageDataGenerator` class from Keras to preprocess and augment the training, validation, and test datasets. The preprocessing steps include:

- Rescaling the pixel values to a range of [0, 1]
- Rotation, width and height shifting, shear, zoom, and horizontal flipping for data augmentation
- Generating data batches for training, validation, and testing

## Model Architecture

The CNN architecture consists of multiple convolutional and pooling layers followed by densely connected layers. The model is compiled using the Adam optimizer and binary cross-entropy loss function.

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## Training the Model

The script trains the CNN using the training dataset and validates the model on a separate validation dataset. The training history, including accuracy and loss metrics, is stored for later analysis.

history = model.fit(train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size)

## Evaluating the Model

The trained model is evaluated on a test dataset, and predictions are generated for sample images. The accuracy of the model is then calculated and compared against predefined ground truth values.

probabilities = model.predict(test_data_gen)
plotImages([test_data_gen[0][0][i] for i in range(5)], [probabilities[i] for i in range(5)])

## Results

The model's performance is assessed based on accuracy, loss, and visual inspection of predicted probabilities for sample images.

For additional details, refer to the code comments and the inline documentation.

Feel free to explore, modify, and provide feedback!

