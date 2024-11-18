# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
Develop a model that can classify images of handwritten digits (0-9) from the MNIST dataset with high accuracy. The model should use a convolutional 
neural network architecture and be optimized using early stopping to avoid overfitting.

## Neural Network Model

Include the neural network model diagram.(http://alexlenail.me/NN-SVG/index.html)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries and  load the dataset

### STEP 2:
Reshape and normalize the data 

### STEP 3:
Create the EarlyStoppingCallback function 

### STEP 4:
Create the convulational model and compile the model

### STEP 5:
Train the model

## PROGRAM

### Name: Vasanth P
### Register Number: 212222240113


```python
import numpy as np
import tensorflow as tf

data_path ="mnist.npz"

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

# reshape_and_normalize
def reshape_and_normalize(images):
  
    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(60000,28,28,1)
    # Normalize pixel values
    images = images/255
    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name: Vasanth P RegisterNumber: 212222240113 \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

# Creating EarlyStoppingCallback function

### START CODE HERE ###

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self,epochs,logs=None):

        # Check if the accuracy is greater or equal to 0.995
        if logs['accuracy'] >= .995:

            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\n\nReached 99.5% accuracy so cancelling training!\n")
            print('Name: Vasanth P  Register Number: 212222240113\n')

#convolutional_model

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    # Define the model
    model = tf.keras.models.Sequential([ 
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ]) 


    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
          
    return model

# Define your compiled (but untrained) model
model = convolutional_model()
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])
```

## OUTPUT

### Reshape and Normalize output

![image](https://github.com/user-attachments/assets/83080d84-0780-4241-9092-8449abe36a92)


### Training the model output

![image](https://github.com/user-attachments/assets/2baa9be8-eede-4a1f-a92a-62d48e993040)



## RESULT
Hence a convolutional deep neural network for digit classification was successfully developed.
