import keras
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sys.exit(0)


def model_1():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 classes

    print(model.summary())
    return model

def resblock(x, kernelsize, filters):
    fx = layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(1, kernelsize, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    return out

def model_2():
    x = Input(shape=(32, 32, 3))
    l1 = resblock(x, 3, 64)
    m1 = layers.MaxPooling2D((2, 2))(l1)
    l2 = resblock(m1, 3, 16)
    m2 = layers.MaxPooling2D((2, 2))(l2)
    f = layers.Flatten()(m2)
    d1 = layers.Dense(32, activation='relu')(f)
    d2 = layers.Dense(10, activation='softmax')(d1)
    return keras.Model(inputs=x, outputs=d2)

def model_3():
    """
    Build CNN model and Perform the following operations:

    1. Flatten the output of our base model to 1 dimension
    2. Add a fully connected layer with 1,024 hidden units and ReLU activation
    3. This time, we will go with a dropout rate of 0.2
    4. Add a final Fully Connected Sigmoid Layer
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, kernel_initializer='he_normal', activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, kernel_initializer='he_normal', strides=1, activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(128, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((4, 4)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal', activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(10, kernel_initializer='glorot_uniform', activation="softmax"))

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    #         model = Sequential()
    #         model.add(self.i_model)
    #         model.add(GlobalAveragePooling2D())
    #         model.add(Dense(128))
    #         model.add(Dropout(0.1))
    #         model.add(Dense(10, activation = 'softmax'))
    #         model.compile(optimizer=SGD(lr=0.0001, momentum=0.99, decay=0.01), loss='categorical_crossentropy', metrics=['acc'])
    return model


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train[:100]
# y_train = y_train[:100]

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = model_3()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlystopper = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)

training = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[earlystopper, reduce_lr],
                    validation_data=(x_test, y_test))

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

f, ax = plt.subplots(1,2, figsize=(12,3))
ax[0].plot(training.history['loss'], label="Loss")
ax[0].plot(training.history['val_loss'], label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Accuracy
ax[1].plot(training.history['accuracy'], label="Accuracy")
ax[1].plot(training.history['val_accuracy'], label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.tight_layout()
plt.show()

