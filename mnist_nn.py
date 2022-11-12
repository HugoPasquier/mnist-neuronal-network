
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras.datasets import mnist

# --- Importing data from mnist database ---

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

# Convert vector to binary matrix
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

def exo1() :
    # Exercice 1
    # Q1
    model = Sequential()    # définition du model (séquentiel = suite de couche)
    model.add(Flatten(input_shape = (28, 28)))  # réduction des image 28x28 en un vecteur (couche entrée)
    # Ajout d'une couche caché pour Q3
    #model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, input_dim = 784, kernel_initializer='normal', activation='softmax')) # couche de traitement à 2 neurones, loi normal pour poids de depart
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])   # optimisation : descencte en gradient

    model.summary()

    # Q2
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)
    scores = model.evaluate(X_test, y_test)
    print("Neural network accuracy: %.2f%%" % (scores[1]*100))

    # Pour améliorer la précisison du réseau sans changer son architecture, on peut faire varier 
    # le nombre d'epochs et le batch_size

    # Q3 
    # L'ajout de la couche caché améliore légérement l'efficacité du réseau
    # Pour Q2, batch_size = 10 -> 97,49%, batch_size = 100 -> 96.74%
    # Pour Q3, batch_size = 10 -> 97,60%, batch_size = 100 -> 96.86%


# Exercice 2
def classic_model() :

    model = Sequential()    # Create model

    # Adding layer
    model.add(Flatten(input_shape = (28, 28)))
    model.add(Dense(10, input_dim = 784, kernel_initializer='normal', activation='softmax'))
    
    # Configure model for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # Training
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)
    
    # Evaluation
    scores = model.evaluate(x_test, y_test)
    print("Neural network accuracy: %.2f%%" % (scores[1]*100))
    #batch_size = 100 ~ 92.80%


def CNN_model() :

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=50) 
    scores = model.evaluate(x_test, y_test)
    print("Neural network accuracy: %.2f%%" % (scores[1]*100))
    #batch_size = 100 -> 99.14%

################ Main ################

classic_model()