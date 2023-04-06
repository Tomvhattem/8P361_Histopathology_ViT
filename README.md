# 8P361 Project AI for Medical Image Analysis
## Group 1
By Tom van Hattem, Marijn de Lange, Koen Vat and Marcus Vroemen

## Libraries and startup
requirements.txt shows all requirements present in the python environment this code was developed on. The main libraries used are tensorflow, tensorflow_addons, keras_tuner and cuda for enabling GPU support .<br />


### Folder overview
├── README.md <br />
├── requirements.txt  <br />
├── ViT.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; — Build and run Vision Transformer<br />
├── ResNet50.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; — Build and run ResNet50 model<br />
├── CNN.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; — Build and run simple CNN model<br />
├── runtensorboard.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; — Script to make opening tensorboard to view learning curves easy<br />
└── model_evaluate.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; — Load model weights to get validation metrics and Kaggle submission file for test metrics

When running the ViT, ResNet or CNN files, the folders logs and model_weights will be created.<br />

### Dataset folder overview
For this project, the PatchCamelyon dataset was saved locally and refered to in the scripts. Make sure to change the paths in every script and use the following folder structure.<br />
├── train+val <br />
&nbsp;&nbsp;&nbsp;&nbsp;├── train <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 0 <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 1 <br />
&nbsp;&nbsp;&nbsp;&nbsp;├── valid <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 0 <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 1 <br />
└── test <br />


## ViT.py
The script ViT.py was based of code from https://keras.io/examples/vision/vit_small_ds/. Make sure to change the path to the location of where your dataset is saved. The first section is responsible for importing libraries and setting the (hyper)parameters. The parameter hp_tuning can be set to True or False to either run the file in tuning mode where hyperparameters are varied to find the ideal combination or normal mode which builds and trains one model with the hyperparameters specified in the top of the script. Important hyperparameters that were tweaked were use_scheduler, LEARNING_RATE, EPOCHS, PATH_SIZE and TRANSFORMER_LAYERS, MINI_EPOCHS and vanilla in the create_vit_classifier() function. The first function get_pcam_generators() is used to load in the dataset as generators since the dataset is too large to be loaded at once. The next functions implement among others the data augmentation, SPT and LSA methods and are called by create_vit_classifier() to build the ViT model. The function run_experiment() is responsible for compining and fitting the model. It also saves the weights of the best model to the folder 'model_weights' and the learning curves to 'logs', which can later on be analysed in tensorboard. For some experiments, the number of training and validation steps were decreased, by for example a factor of 5, to train with so-called 'mini-epochs'. The model can be compiled with either a constant learning rate, or use the warmup cosine learning rate scheduler by setting use_scheduler on True, which then calls the function WarmUpCosine(). This makes sure that the learning rate increases from warmup_learning_rate to learning_rate_base in warmup_steps number of steps before it gradually decreases with cosine decay. The tuner can be customized at the bottom and hyperparameter choises can be changed within the keras_tuner_build() function.

## ResNet.py
The file ResNet.py contains many of the same features as ViT.py with some exceptions. In this file, only the parameters hp_tuning, use_scheduler, LEARNING_RATE and EPOCHS were changed. In this file, the model is build by using the ResNet50 model from the tensorflow.keras.applications.resnet50 library. The top was included and connected to a single sigmoid activated node in order to make binairy classifications. Make sure again to change the specified paths to your personal folder were the dataset is stored.

## CNN.py
CNN.py contains basically the same code as the previous files by without tuner and a far simpler model containing only two convolutional, two max pooling and three dense layers connected to one output node.




