from ViT_Tomvh_copy_minas_ithil_final import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread

from tensorflow.keras.models import model_from_json


model = create_vit_classifier(vanilla=False)
optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        #keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

model_name = 'VIT_vanilla'
checkpoint_filepath = 'model_weights\\' + model_name + '\\'
BATCH_SIZE = 256

model.load_weights(checkpoint_filepath)
#_, accuracy = model.evaluate(test_gen, batch_size=BATCH_SIZE)
#print(f"Test accuracy: {round(accuracy * 100, 2)}%")


# open the test set in batches (as it is a very big dataset) and make predictions
TEST_PATH = 'C:/Data/test/test/' 
test_files = glob.glob(TEST_PATH + '*.tif')

submission = pd.DataFrame()

file_batch = 5000
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
submission.to_csv('submission_vit_final_og.csv', index = False, header = True)