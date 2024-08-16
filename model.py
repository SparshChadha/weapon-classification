import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
IMAGE_SIZE        = 360
BATCH_SIZE        = 32
CHANNELS          = 3
EPOCHS            = 10
TRAIN_SIZE        = 0.8
TEST_SIZE         = 0.1
VALIDATION_SIZE   = 0.1
SHUFFLE_SIZE      = 1000
SEED              = 72

# Load and preprocess dataset from directory
def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "data2",
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    return dataset

# Split dataset into training, validation, and testing sets
def get_dataset_partition_tf(ds, train_split, val_split, shuffle=True, shuffle_size=SHUFFLE_SIZE):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=SEED)
    
    train_size = int(train_split * ds_size)
    val_size   = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, test_ds, val_ds

# Define preprocessing and data augmentation layers
def build_preprocessing_layers():
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1.0 / 255),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ])

    return resize_and_rescale, data_augmentation

# Build the CNN model
def build_model(input_shape, n_classes, resize_and_rescale, data_augmentation):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),


        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # Adding dropout for regularization
        layers.Dense(n_classes, activation='softmax')
    ])

    return model

# Compile and train the model
def compile_and_train(model, train_ds, val_ds, epochs, batch_size):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=val_ds
    )

    return history

# Evaluate the model
def evaluate_model(model, test_ds):
    scores = model.evaluate(test_ds)
    return scores

# Plot training and validation accuracy and loss
def plot_training_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label="Training Accuracy")
    plt.plot(range(epochs), val_acc, label="Validation Accuracy")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label="Training loss")
    plt.plot(range(epochs), val_loss, label="Validation loss")
    plt.legend(loc='lower right')
    plt.title('Training and Validation loss')
    plt.show()

# Predict and display results
def predict_and_display(model, test_ds):
    for images_batch, labels_batch in test_ds.take(1):
        first_image = images_batch[0].numpy().astype('uint8')
        first_label = labels_batch[0].numpy()

        print("First image to predict:")
        plt.imshow(first_image)
        print("Actual label: ", class_names[first_label])

        batch_prediction = model.predict(images_batch)
        print("Predicted label: ", class_names[np.argmax(batch_prediction[0])])

def predict_single_image(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Display predictions for a batch of images
def display_predictions(model, test_ds, class_names):
    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))

            predicted_class, confidence = predict_single_image(model, images[i], class_names)
            actual_class = class_names[labels[i]]

            plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence}%")
            plt.axis('off')
    plt.show()

# Save the model
def save_model(model, save_path):
    model.save(save_path + ".keras")

# Execution
dataset = load_dataset()
class_names = dataset.class_names

train_ds, test_ds, val_ds = get_dataset_partition_tf(dataset, TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE)
train_ds = train_ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale, data_augmentation = build_preprocessing_layers()

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)  # Correct input shape
model = build_model(input_shape, len(class_names), resize_and_rescale, data_augmentation)

print(model.summary())

history = compile_and_train(model, train_ds, val_ds, EPOCHS, BATCH_SIZE)

scores = evaluate_model(model, test_ds)

plot_training_history(history, EPOCHS)

predict_and_display(model, test_ds)

display_predictions(model, test_ds, class_names)

model_version = max([int(i) for i in os.listdir("../research paper/model") + [0]]) + 1
save_model(model, f"../research paper/model/{model_version}")
