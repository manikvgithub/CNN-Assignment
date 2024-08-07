# Project Name
This project is about building a CNN Model which can accurately detect Melanoma.  Melanoma is a type of skin cancer that can be deadly if not detected early.  Therefore, early detection can help dermatologists in reducing the manual efforts needed for its diagnosis and cure.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)


## General Information
- The assignment aims to accurately detect the presence of melanoma disease based on the images provided.  Convolutional Neural Networks (CNN) methodology is applied to scan thru a varity of images provided to identify if it is of melanoma type.
- The data set contained a total of 2239 images for training and 118 for testing, divided into 9 classes. Each class representing a type of skin cancer. 
- Based on the above information CNN model is built trying to predict whether the given image is of type melanoma or not.

Approach:
   - From the given dataset of images, two folders were given - Train and Test.
   - Train folder contained 2239 images whereas the test contained 118 images
   - The Train images were first split into Train Data Set and Validation Data Set on a 80%-20% ratio.  This resulted in using  1792 images for training and 447 for validation respectively
    - The images were spead across 9 classes. One image from each class for both training and validation was displayed to confirm we have representation from each class
    - The image was provided with the tensor of shape (32, 180, 180, 3).  Each was a batch of 32 images of shape 180x180x3 (the last dimension refers to color channels RGB). The label batch itself was a tensor of the shape (32,).
    - The images were Rescaled using Keras to have standardized values between 0 and 1. This is primarily done to help with CNN modeling.
    - The first model architecture used 2 convolutional layers with 32 and 64 labels, relu activation function, 2x2 maxpooling and dropouts.
      - The model was fit with 20 epochs and provided 86% accuracy for training and 44% for Validation sets.  The model result was clearly an Ovefitting scenario.
      - To help reduce the overfitting, Augumenation approach was applied.  
      - For the augumentation exercise, the model architecture was modified to apply Batch Normalization in each Convolutional layer, using L2 regularization function at the FC layer and using droputs.  Also, data was augumented with  RandomFlip, RandomRotation and RandomCrop functions.
      - The model architecture had 4 convolutional layers each having 32, 64, 128 and 256 layers within them.  Batch Normalization and maxpooling was incorporated in each layer along with L2 regularization technique with 0.01 value at the end (in the Dense layer)
      - After fitting the model, got an accuracy score of 52.7% on training and 47.7% on Validation. The amount of variation had reduced between these two data sets but there was still an Overfitting problem.
      - As a next step, looked at the possibility of having Class Imbalance in the provided data set
      - The distribution of sample images provided in the training set had too much variation as can be seen in the table below:
    
        Class Name                      Number of Images    
        ===================================================
        actinic keratosis               114 (5.092%)
        basal cell carcinoma            376 (16.793%)
        dermatofibroma                  95 (4.243%)
        melanoma                        438 (19.562%)
        nevus                           357 (15.945%)
        pigmented benign keratosis      462 (20.634%)
        seborrheic keratosis            77 (3.439%)
        squamous cell carcinoma         181 (8.084%)
        vascular lesion                 139 (6.208%)
         
       - To address this issue, 500 images were added to each class by creating an Output folder for each class.
       - Post this augumenation exercise, the number of images under each class was as follows:

        Class Name                      Number of Images    
        ===================================================
        actinic keratosis               614 
        basal cell carcinoma            876 
        dermatofibroma                  595 
        melanoma                        938 
        nevus                           857 
        pigmented benign keratosis      962 
        seborrheic keratosis            577 
        squamous cell carcinoma         681 
        vascular lesion                 639 

        - After these sample images were infused into the dataset, the train and validation data sets were split with 80%-20% ratio again resulting in using 5392 and 1347 images for these two categories respectively
         - Next the model was built with these datasets and the architecture now had 3 convolutional layers with 16,32 and 64 layers with relu activation function and maxpooling
         - A drop out of 20% was added at the end of the network 
         - The model gave an accuracy of 95.3% for Training and and 80.9% for validation. Though the overfitting issue was still there, the accuracy levels have increased significantly.
         - Finally, this model was applied to the test dataset, after applying Rescaling on it.  The accuracy thus obtained was 38.35%
         - Throughout the exercise, the adam optimizer was used along with SparseCategoricalCrossentropy loss function.
        
## Conclusions
- Throughout the CNN model building exercise for this project, we observed that at each step, the level of accuracy improved and the difference in accuracy between training and validation datasets reduced, which proved that the approach being taken while building the model has a significant impact
- The next important observation is the usage of augumenation layer in the model and its impact on the outcome of the model. Though the Overfitting problem was there, the difference in accuracy between validation and training layers narrowed to a large extent - from 38% difference to just 5%.
- The Class Imbalance issue identified with the given samples in the 9 classes and the addition of 500 images to each class helped in improving the accuracy levels further. The accuracy values for the final model yeilded 95.3% for Training and 80.9% for validation.
- The test data set yeilded a 38.35% accuracy.
- Though the overfitting issue was seen throughout the modeling exercise, improvement in accuracy values were signifcant at the end.



## Technologies Used
- Google Colab
- Python           3.10.12
- TensorFlow       2.17.0
- Numpy            1.26.4
- Pandas           2.1.4
- Keras            3.4.1
- Augmentor        0.2.12


## Acknowledgements
References 
- Course materials from upGrad curriculam
- stackoverflow and geekforgeek websites for examples
- Tensorflow Documentation	https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
                            https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling
- Optimizers	            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
- Categorical cross entropy	https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy
- Cross Entropy vs. Sparse Cross Entropy: When to use one over the other	https://stats.stackexchange.com/questions/326065/cross-entropy-vs-sparse-cross-entropy-when-to-use-one-over-the-other



## Contact
Manikandan Krishnamurthy Vembu (https://github.com/manikvgithub)


