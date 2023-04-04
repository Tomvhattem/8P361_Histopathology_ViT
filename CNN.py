"""
TU/e Biomedical Engineering
Group 01 For 8P361 Project AI for medical image analysis

Simple CNN
"""
epochs = 50

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras


#Fix the random seed
from tensorflow.random import set_seed
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(42)
   set_seed(42)
   np.random.seed(42)

#make some random data
reset_random_seeds()


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

#Learning rate scheduler class:
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
        return tf.where(
            step > self.total_steps, 1e-6, learning_rate, name="learning_rate"
        )

#Data generators:
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir,'train+val', 'train')
     valid_path = os.path.join(base_dir,'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

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


#Model structure:
def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     #Add the layers:
     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(64, activation='relu'))
     model.add(Dense(64, activation='relu'))
     model.add(Dense(1, activation = 'sigmoid'))

     #Initialise scheduled learning rate
     scheduled_lrs = WarmUpCosine(
         learning_rate_base=0.001,
         total_steps=total_steps,
         warmup_learning_rate=0.0,
         warmup_steps=warmup_steps,
     )

     # compile the model
     model.compile(SGD(learning_rate=scheduled_lrs, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model

# get the data generators
train_gen, val_gen, test_gen = get_pcam_generators('C:\\Data\\')
total_steps = int((train_gen.n / train_gen.batch_size) * epochs) #train_gen.n to num_elements for dataset
warmup_epoch_percentage = 0.10
warmup_steps = int(total_steps * warmup_epoch_percentage)

# get the model
model = get_model()

# save the model and weights
model_name = 'Cnn_sched_50_epochs'
model_filepath = 'model_weights\\' + model_name + '.json'
weights_filepath = 'model_weights\\' + model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
#checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_filepath = 'model_weights\\' + model_name + '\\'

#Setup the callbacks
checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint_callback, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=epochs,
                    callbacks=callbacks_list)

#Load the best weights for the evaluation.
model.load_weights(checkpoint_filepath)

_, accuracy  = model.evaluate(test_gen, batch_size=train_gen.batch_size)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

