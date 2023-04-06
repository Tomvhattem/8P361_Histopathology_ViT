#%%
import ViT
import ResNet 
import CNN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
from tensorflow import keras
from matplotlib.pyplot import imread

# change path to path folder with dataset
data_path = 'C:\\Data\\'

# Set seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)

IMAGE_SIZE = 96
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    RESCALING_FACTOR = 1. / 255

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

# change path to path folder with dataset
train_gen, val_gen = get_pcam_generators(data_path)

#%% Get validation metrics from saven model weights
model_name='' #name of model weights
MODEL_SELECTER=0 #0 for SL-ViT, 1 for vanilla ViT, 2 for resnet and 3 for the simple cnn

metrics = ['accuracy', 'AUC']
for metric in metrics:
    if MODEL_SELECTER==0:
        model=ViT.create_vit_classifier(vanilla=False)
    elif MODEL_SELECTER==1:
        model=ViT.create_vit_classifier(vanilla=True)
    elif MODEL_SELECTER==2:
        model=ResNet.build_model()    
    elif MODEL_SELECTER==3:
        model=CNN.get_model()

    model.compile(metrics=metric)

    model.summary()

    checkpoint_filepath = 'model_weights\\' + model_name + '\\'
    model.load_weights(checkpoint_filepath)

    loss, metric_val = model.evaluate(val_gen, batch_size=32)
    
    print(model_name)
    print("Validation loss: {:.3f}".format(loss))
    print("Validation "+metric+f": {round(metric_val * 100, 2)}%")
    


#%% KAGGLE SUBMISSION
# open the test set in batches (as it is a very big dataset) and make predictions
TEST_PATH = data_path+'test\\' 
test_files = glob.glob(TEST_PATH + '*.tif')

submission = pd.DataFrame()

file_batch = 500
max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):

    print('Indexes: %i - %i'%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})


    # get the image id 
    test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
    test_df['image'] = test_df['path'].map(imread)
    
    
    K_test = np.stack(test_df['image'].values)
    
    # apply the same preprocessing as during draining
    K_test = K_test.astype('float')/255.0
    
    predictions = model.predict(K_test)
    
    #test_df['label'] = abs(predictions[:,0]) 
    test_df['label'] =predictions[:,0]
    #test_df['label']=(test_df['label']-test_df['label'].min())/(test_df['label'].max()-test_df['label'].min())
    submission = pd.concat([submission, test_df[['id', 'label']]])

# save your submission
submission.head()
submission.to_csv('submission_'+model_name+'.csv', index = False, header = True)

