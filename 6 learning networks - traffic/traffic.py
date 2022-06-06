import cv2
import numpy as np
import os
import sys
import tensorflow as tf
"""
# note:
# tensorflow does not install in Thonny using the bundled Python 3.7.9, update to a different Python version
# when using a different Python version, you need to re-install any packages you need for that version:
# - opencv-python
# - scikit-learn
# - tensorflow
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, shuffle=True)
    
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=history.history['val_loss'], name="val_loss"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter( y=history.history['loss'], name="loss"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter( y=history.history['val_accuracy'], name="val_accuracy"),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter( y=history.history['accuracy'], name="accuracy"),
        secondary_y=True
    )
    # Add figure title
    fig.update_layout(
        title_text="Loss/Accuracy of NNS Model"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>primary</b> Loss", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> Accuracy", secondary_y=True)
    
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        fig.write_image(filename + '.png')
        model.save(filename)
        print(f"Model saved to {filename}.")
    else:
        fig.show()


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    # images is a list of all the images in the data directory
    images = []
    
    # labels of integers representing categories
    # NUM_CATEGORIES
    labels = []
    
    # https://www.geeksforgeeks.org/os-walk-python/
    for root, folders, files in os.walk(data_dir):        
        for file in files:
            # in case we have other files in our data_dir
            if file.endswith('.ppm'):
                
                # os.walk will walk through all subfolders on its own, so limit it to the number of categories
                try:
                    if int(os.path.basename(root)) > int(NUM_CATEGORIES):
                        print("skipping folder",os.path.basename(root))
                        continue
                except:
                    print("differently named folder found:",os.path.basename(root))
                    
                # for platform independency, use os.sep and os.path.join as needed
                img = cv2.imread(os.path.join(root, file))    
                # Use cv2 to read each image as a numpy.ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3 
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                # append to the lists
                images.append(img)
                labels.append(int(os.path.basename(root)))
                
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers
    # https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning-86737aedc2ab
    model = get_model_snel_9()
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    return model


def get_model_snel_0():
    """   
    %run traffic.py gtsrb [model.c-m-relu5-c-m-f-d5-s]
    333/333 - 4s - loss: 0.1174 - accuracy: 0.9703 - 4s/epoch - 13ms/step
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Dense: relu layers
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Flatten units
        tf.keras.layers.Flatten(),        
        # Dropout layer
        tf.keras.layers.Dropout(0.5),       
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_1():
    """
    %run traffic.py gtsrb [model.c-m-c-f-d3-s]
    333/333 - 1s - loss: 0.5208 - accuracy: 0.9157 - 1s/epoch - 3ms/step
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),   
        # Dropout layer
        tf.keras.layers.Dropout(0.3),
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_2():
    """ 
    %run traffic.py gtsrb [model.c-a-c-f-d3-s]
    333/333 - 4s - loss: 0.0860 - accuracy: 0.9788 - 4s/epoch - 13ms/step
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Avg-pooling layer
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),     
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_3():
    """
    %run traffic.py gtsrb [model.c-a22-c-f-d3-s]
    333/333 - 4s - loss: 0.0988 - accuracy: 0.9791 - 4s/epoch - 13ms/step
    ondanks strides en padding nauwelijks verschil in accuratesse, iets meer verlies
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer with strides and padding
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),strides=(2, 2), padding="valid"),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),        
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_4():
    """
    %run traffic.py gtsrb [model.c-a22-r1-c-f-d3-s]
    333/333 - 2s - loss: 0.2542 - accuracy: 0.9588 - 2s/epoch - 5ms/step
    ondanks hidden layer geen vooruitgang in accuratesse, meer verlies
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer with strides and padding
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),strides=(2, 2), padding="valid"),
        # Dense: relu layers
        tf.keras.layers.Dense(128, activation="relu"),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),        
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_5():
    """
    %run traffic.py gtsrb [model.c-a-r2-c-f-d3-s]
    333/333 - 2s - loss: 0.1869 - accuracy: 0.9611 - 2s/epoch - 7ms/step
    extra hidden layer heeft weinig invloed
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer with strides and padding
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),strides=(2, 2), padding="valid"),
        # Dense: relu layers
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),        
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_6():
    """
    %run traffic.py gtsrb [model.c-m22-r2-c-f-d3-s]
    333/333 - 3s - loss: 0.1619 - accuracy: 0.9747 - 3s/epoch - 8ms/step
    MaxPooling2D in plaats van AveragePooling2D 
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer with strides and padding
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding="valid"),
        # Dense: relu layers
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),        
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_7():
    """
    %run traffic.py gtsrb [model.c-m22-r3-c-f-d3-s]
    333/333 - 3s - loss: 0.1543 - accuracy: 0.9695 - 3s/epoch - 9ms/step
    extra hidden layer leidt niet tot betere accuracy
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer with strides and padding
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding="valid"),
        # Dense: relu layers
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_8():
    """
    %run traffic.py gtsrb [model.c-m22-r1-d3-r4-f-d3-s]
    333/333 - 23s - loss: 0.1410 - accuracy: 0.9670 - 23s/epoch - 69ms/step
    extra hidden layer overkill leidt niet tot betere accuracy
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer with strides and padding
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding="valid"),
        # Dense: initial relu layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 32, activation="relu"),
        # Dropout layer
        tf.keras.layers.Dropout(0.3),
        # Dense: relu layers
        tf.keras.layers.Dense(NUM_CATEGORIES * 16, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES * 8, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES * 4, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES * 2, activation="relu"),
        # Flatten units
        tf.keras.layers.Flatten(),       
        # Dropout layer
        tf.keras.layers.Dropout(0.3),
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_snel_9():
    """   
    %run traffic.py gtsrb [model.c-m-r5-c-m-f-d5-s]
    333/333 - 4s - loss: 0.1174 - accuracy: 0.9703 - 4s/epoch - 13ms/step
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Dense: relu layers
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Flatten units
        tf.keras.layers.Flatten(),        
        # Dropout layer
        tf.keras.layers.Dropout(0.5),       
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


def get_model_gh1():
    """
    %run traffic.py gtsrb [model.c-m-d2-c-m-d2-f-r1-d2-s]
    333/333 - 2s - loss: 0.0998 - accuracy: 0.9724 - 2s/epoch - 5ms/step
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer. learns 32 filters using 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Adding dropout to remove 20% random neurons from the net on every iteration
        tf.keras.layers.Dropout(0.2),
        # Convolutional layer. learns 64 filters using 3x3 kernel
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Dropout layer
        tf.keras.layers.Dropout(0.2),
        # Flatten units
        tf.keras.layers.Flatten(),
        # Adding a hidden layer with 128 neurons
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.2),
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    return model


def get_model_gh2():
    """
    %run traffic.py gtsrb [model.c-m-f-d2-rd2-r3-s]
    333/333 - 2s - loss: 0.1824 - accuracy: 0.9547 - 2s/epoch - 7ms/step
    """
    model = tf.keras.models.Sequential([
        # Learn 32 filters using a  grid of 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        # Flatten units
        tf.keras.layers.Flatten(),
        # Dropout layer
        tf.keras.layers.Dropout(0.2),
        # Hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 32, activation="relu"),
        # Dropout layer
        tf.keras.layers.Dropout(0.2),
        # Another hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 16, activation="relu"),
        # Another hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 8, activation="relu"),
        # Another hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 4, activation="relu"),
        # Add an output (softmax) layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model


if __name__ == "__main__":
    main()
