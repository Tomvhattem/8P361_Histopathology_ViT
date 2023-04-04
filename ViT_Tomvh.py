
"""
TU/e Biomedical Engineering
Group 01 For 8P361 Project AI for medical image analysis

Code partially based on ViT For small datasets
https://arxiv.org/abs/2112.13492 by S.H Lee

Visual Transformer with Shifted Patch Tokenization and Locality Self-Attention for small
datasets.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import math 
import os
import sys
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import keras_tuner
from datetime import datetime

model_name = 'VIT_Final'
use_scheduler = True #Set to true for final model
hp_tuning = False   #Tuning parameter which (de)activates the keras tuner. False for final model

#Sets Random Seet
SEED = 42
keras.utils.set_random_seed(SEED)

#Data Preprocessing
NUM_CLASSES = 1
IMAGE_SIZE = 96
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

#Hyperparameters
# OPTIMIZER
if use_scheduler == True:
    LEARNING_RATE = 1e-3 
else:
    LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.0001

BATCH_SIZE = 32
buffer_size = 512
EPOCHS = 500
MINI_EPOCHS = 1 #Decrease the stepsize to get smaller epochs for faster training time

PATCH_SIZE = 16
TRANSFORMER_LAYERS = 11
NUM_PATCHES = (IMAGE_SIZE//PATCH_SIZE)**2

LAYER_NORM_EPS = 1e-6

PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
MLP_HEAD_UNITS = [2048, 1024]

def get_pcam_generators(base_dir, IMAGE_SIZE, train_batch_size=32, val_batch_size=32):
    """ Image generator class from tensorflow keras that generates batches of images."""

    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')
    test_path = os.path.join(base_dir, 'test', 'test')
	 
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
    return train_gen, val_gen

train_gen, val_gen = get_pcam_generators('C:\\Data\\',BATCH_SIZE,BATCH_SIZE) #,IMAGE_SIZE,SEED

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

class ShiftedPatchTokenization(layers.Layer):
    """Class for Shifted Patch Tokenization
    shifts an input image spatially in several direction"""
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)
    
class PatchEncoder(layers.Layer):
    """"Patch Encoder class"""
    def __init__(
        self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches

class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    """MultiHeadAttention Locality Self Attention, at its core contains
    diagonal masking and a learnable temperature."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

def mlp(x, hidden_units, dropout_rate):
    """Multilayer Perceptron"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Build the diagonal attention mask
diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

def create_vit_classifier(vanilla=False):
    """Function to create the tensorflow model object including all architecture elements"""
    inputs = layers.Input(shape=INPUT_SHAPE)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder()(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    # Classify outputs.
    sigmoid = layers.Dense(NUM_CLASSES,activation='sigmoid')(features) #sigmoid niet softmax want das voor multiclass
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=sigmoid)
    return model

#The keras tuner model
def keras_tuner_build(hp):
    """Specific funtion for the keras tuner with only hp, hyperparameter as input
    and return a compiled model with variable hyperparameters"""
    TRANSFORMER_LAYERS = hp.Int("Transformer_layers", min_value=4, max_value=16)
    PATCH_SIZE = hp.Int("Patch Size", min_value=4, max_value=16)
    
    NUM_PATCHES = (IMAGE_SIZE//PATCH_SIZE)**2
    model = create_vit_classifier(vanilla=False)
 
    total_steps = int((train_gen.n / BATCH_SIZE) * EPOCHS) #train_gen.n to num_elements for dataset
    warmup_epoch_percentage = 0.10
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    # Select optimizer    
    hp_optimizer = hp.choice("LR_optimizer",['AdamW','RMSProp'])
    if hp_optimizer == 'AdamW':
            optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif hp_optimizer == 'RMSProp':
            optimizer = tfa.optimizers.experimental.RMSprop(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
       


    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            #keras.metrics.SparseCategoricalAccuracy(name="AUC"),        
        ],
    )
    return model

class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate scheduler with cosine decay and warmup"""
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
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

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
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

def run_experiment(model):
    """Function that executes training and evaluation (excluding kaggle)
    of the final model/custom experiments."""
    total_steps = int((train_gen.n / BATCH_SIZE) * EPOCHS) #train_gen.n to num_elements for dataset
    warmup_epoch_percentage = 0.10
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    if use_scheduler == True:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY
        )
    elif use_scheduler == False:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),   
        ],
    )

    train_steps = train_gen.n // train_gen.batch_size // MINI_EPOCHS
    val_steps = val_gen.n // val_gen.batch_size // MINI_EPOCHS
   
    model_filepath = 'model_weights\\' + model_name + '.json'
    weights_filepath = 'model_weights\\' + model_name + '_weights.hdf5'
    model_loc = 'model_weights\\' + model_name
    checkpoint_filepath = 'model_weights\\' + model_name + '\\'

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    tensorboard = TensorBoard(os.path.join('logs', (model_name)))

    history = model.fit(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=EPOCHS,
                        callbacks=[tensorboard,checkpoint_callback])
    
    model.load_weights(checkpoint_filepath)

    #Evaluation on validation data, as test validation can only be done through kaggle.
    _, accuracy  = model.evaluate(val_gen, batch_size=BATCH_SIZE)
    print(f"Val accuracy: {round(accuracy * 100, 2)}%")

    return history



if __name__ == '__main__':
    if hp_tuning == True:
        print('starting tuning process')
        """Tuning with keras tuner, tunable hyperparameters given in the keras
        tuner function: keras_tuner_build."""

        train_steps = train_gen.n // train_gen.batch_size // MINI_EPOCHS
        val_steps = val_gen.n // val_gen.batch_size // MINI_EPOCHS

        tuner = keras_tuner.BayesianOptimization(
            keras_tuner_build,
            objective=['val_loss','val_accuracy'],
            max_trials=3,
            seed=SEED,
            executions_per_trial=1,
            overwrite=True,
            project_name = 'Main_ViT4' )
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(train_gen, epochs=30,
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps,
                        validation_data=(val_gen),
                        callbacks=[stop_early])
        best_model = tuner.get_best_models()[0]
        best_hps=tuner.get_best_hyperparameters()[0]

        tuner.search_space_summary()

    else:
        #Run experiments with the Shifted Patch Tokenization and
        #Locality Self Attention modified ViT
        vit_sl = create_vit_classifier(vanilla=False)
        
        history = run_experiment(vit_sl)

