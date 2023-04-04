'''
ResNet notes
- Top layer with 1 class, sigmoid, run until converged
- Learningrateschedule, data augmentation, adamW, binairycrossentropy
- No tuning, no regularization (with StochasticDepth)

Tutorials/sources:
- https://www.tensorflow.org/guide/keras/transfer_learning
- https://www.tensorflow.org/tutorials/images/transfer_learning
- https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth
- https://keras.io/api/optimizers/
- https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/ 

'''

import numpy as np
import matplotlib.pyplot as plt
import math 
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
#import keras_tuner #pip install -q -U keras-tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import keras_tuner


# Set seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)

# Toggle hp_tuning to enable tune mode
hp_tuning = True
use_scheduler = False

"""
## Set (hyper)parameters
"""
# Image size and #classes from PCAM dataset
NUM_CLASSES = 1
IMAGE_SIZE = 96
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

MINI_EPOCHS = 1
WEIGHT_DECAY = 0.0001

if use_scheduler == True:
    LEARNING_RATE = 1e-3 
else:
    LEARNING_RATE = 1e-4

# TRAINING
EPOCHS = 50 

"""
## Get the data from generators
"""

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val','valid')
    
    # instantiate data generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary')
    val_gen = datagen.flow_from_directory(valid_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=val_batch_size,
                                            class_mode='binary')
    test_gen = datagen.flow_from_directory(valid_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=val_batch_size,
                                            class_mode='binary')
    return train_gen, val_gen, test_gen

train_gen, val_gen, test_gen = get_pcam_generators('C:\\Data\\')

"""
Build the ResNet model
"""


# Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor = 0.2, width_factor=0.2),
    ],
    name='data_augmentation'
)

# ResNet50 base model
base_model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=INPUT_SHAPE,
    classes = NUM_CLASSES,
    classifier_activation = 'sigmoid'
)

def build_model():
    input = layers.Input(shape=INPUT_SHAPE)
    # Augment data
    augmented = data_augmentation(input)
    # Base model
    output = base_model(augmented, training=True)
    # Create model
    model = Model(input, output)
    return model

"""Compile, train, and evaluate the model"""

# Create learning rate schedule function
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr+1e-6)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
            print(learning_rate)
        return tf.where(
            step > self.total_steps, 1e-6, learning_rate, name="learning_rate" 
        )
    
    def get_config(self):
        config = {
        'learning_rate_base': self.learning_rate_base,
        'total_steps': self.total_steps,
        'warmup_learning_rate': self.warmup_learning_rate,
        'warmup_steps': self.warmup_steps
        }
        return config


def run_experiment(model, model_name="ResNet50_test", hp_tuning=False, hp=0):
    """"Running the experiment for the final run and experiments"""
    # Define the model checkpoint and Tensorboard callbacks
    checkpoint_filepath = 'model_weights\\' + model_name + '\\'
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    tensorboard = TensorBoard(os.path.join('logs', (model_name)))

    # define the # steps per epoch and train the model
    train_steps = train_gen.n//train_gen.batch_size //MINI_EPOCHS
    val_steps = val_gen.n//val_gen.batch_size //MINI_EPOCHS

    warmup_epoch_percentage = 0.10
    total_steps = train_steps*EPOCHS
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    # Setup learning rate scheduler
    
    if hp_tuning==False:
        scheduled_lrs = WarmUpCosine(
            learning_rate_base=LEARNING_RATE,
            total_steps=int(total_steps),
            warmup_learning_rate=0.0,
            warmup_steps=int(warmup_steps),
        )
            # Compile the model with optimizer and loss function
        if use_scheduler == True:
            model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY), 
                    loss=keras.losses.BinaryCrossentropy(from_logits=False), 
                    metrics=['accuracy'])
        elif use_scheduler == False:
            model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY), 
                    loss=keras.losses.BinaryCrossentropy(from_logits=False), 
                    metrics=['accuracy'])

        
    # Alter lr when tuning hyperparameters
    else:
        hp_learning_rate = hp.Float("lr", [1e-2, 1e-4])
        warmup_epoch_percentage = 0.10
        warmup_steps = int(total_steps * warmup_epoch_percentage)
        scheduled_lrs = WarmUpCosine(
            learning_rate_base=hp_learning_rate,
            total_steps=int(total_steps),
            warmup_learning_rate=hp_learning_rate*0.1,
            warmup_steps=warmup_steps,
        )
        hp_optimizer = hp.choice("LR_optimizer",['AdamW','RMSProp'])
        if hp_optimizer == 'AdamW':
            optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
        elif hp_optimizer == 'RMSProp':
            optimizer = tfa.optimizers.experimental.RMSprop(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
       
        model.compile(optimizer=optimizer,
                loss=keras.losses.BinaryCrossentropy(from_logits=False), 
                metrics=['accuracy'])
        
        return model


    history = model.fit(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=EPOCHS,
                        callbacks=[tensorboard,checkpoint_callback])
    
    
    # Print latest metrics
    print("Test Accuracy: ",history.history['accuracy'])
    print("Test Loss: ",history.history['loss'])
    print("Val Accuracy: ",history.history['val_accuracy'])
    print("Val Loss: ",history.history['val_loss'])
    
    # Load weights of best best version of the model
    model.load_weights(checkpoint_filepath)
    
    _, accuracy = model.evaluate(test_gen, batch_size=32)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    
    return history


def keras_tuner_build(hp):
    """keras tuner model with hyperparameter as hp as only input, and returning 
    a compiled keras model"""
    heckpoint_filepath = 'model_weights\\' + model_name + '\\'
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    tensorboard = TensorBoard(os.path.join('logs', (model_name)))

    # define the # steps per epoch and train the model
    train_steps = train_gen.n//train_gen.batch_size // MINI_EPOCHS
    val_steps = val_gen.n//val_gen.batch_size  // MINI_EPOCHS

    warmup_epoch_percentage = 0.10
    total_steps = train_steps*EPOCHS
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    # Setup learning rate scheduler

    #hp_learning_rate = hp.Float("lr", [1e-3, 1e-4])
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    warmup_epoch_percentage = 0.10
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=hp_learning_rate,
        total_steps=int(total_steps),
        warmup_learning_rate=hp_learning_rate*0.1,
        warmup_steps=warmup_steps,
    )
    hp_optimizer = hp.Choice("LR_optimizer",['AdamW','RMSProp'])
    if hp_optimizer == 'AdamW':
        optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
    elif hp_optimizer == 'RMSProp':
        optimizer = tfa.optimizers.RMSprop(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
    
    model.compile(optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=False), 
            metrics=['accuracy'])
    
    return model


if __name__ == '__main__':
    # fetch date to incorporate in model_name
    now = datetime.now()
    dt_string = now.strftime("%d-%m")
        
    if hp_tuning == False:
        # Build and  run model with preset hyperparameters
        model = build_model()
        history = run_experiment(model=model, model_name=('ResNet50_FinalV3_50epoch'), hp_tuning=False)
    
    elif hp_tuning == True:
        # Start tuner to find best hyperparameters
        print('starting tuning process')
        tuner = keras_tuner.BayesianOptimization(
            keras_tuner_build,
            objective='val_accuracy',
            max_trials=6,
            seed=SEED,
            executions_per_trial=1,
            overwrite=True,
            directory='my_dir',
            project_name = 'ResNet50_tune_'+dt_string )
        tuner.search(train_gen, epochs=3, validation_data=(val_gen))
        best_model = tuner.get_best_models()[0]
        best_hps=tuner.get_best_hyperparameters()[0]
        tuner.search_space_summary()

        # Finally run best model
        # besth_model = build_model(best_hps[0], hp_tuning=True) #Error 0 dpes mpt exist
        # history = run_experiment(besth_model)



