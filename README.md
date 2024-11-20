# Melanoma Detection
- Melanoma Detection refers to the process of identifying melanoma, a type of skin cancer, from images or clinical data, often using machine learning or image processing techniques. 
- Melanoma is one of the most dangerous forms of skin cancer and can spread to other parts of the body if not detected early. Early detection is critical for successful treatment, and as such, melanoma detection has become a key area of research, especially with the rise of deep learning and artificial intelligence (AI).


## Table of Contents
* [General Info](#general-information)
* [Problem statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Provide general information about your project here.
- What is the background of your project?
- What is the business probem that your project is trying to solve?
- What is the dataset that is being used?

## Problem statement
- To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

### Business Understanding

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

* Actinic keratosis
* Basal cell carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented benign keratosis
* Seborrheic keratosis
* Squamous cell carcinoma
* Vascular lesion

### Business Goal:

We have to build a multiclass classification model using a custom convolutional neural network in TensorFlow. 

### Business Risk:

- Predicting a incorrect class of skin cancer

## Project Pipeline
- Understand the data → Define the path loading train and test images 
- Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
- Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
- Model Building & training : 
  - Create a CNN model, which can accurately detect 9 classes present in the dataset. 
  - While building the model, rescale images to normalize pixel values between (0,1).
  - Choose an appropriate optimiser and loss function for model training
  - Train the model for ~20 epochs
  - Write your findings after the model fit. You must check if there is any evidence of model overfit or underfit.
- Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
- Model Building & training on the augmented data :
  - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
  - Choose an appropriate optimiser and loss function for model training
  - Train the model for ~20 epochs
  - Write your findings after the model fit, see if the earlier issue is resolved or not?
- Class distribution: Examine the current class distribution in the training dataset 
  - Which class has the least number of samples?
  - Which classes dominate the data in terms of the proportionate number of samples?
- Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
- Model Building & training on the rectified class imbalance data :
  - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
  - Choose an appropriate optimiser and loss function for model training
  - Train the model for ~30 epochs
  - Write your findings after the model fit, see if the issues are resolved or not?

## Conclusions
Multiple models were created to bring the models to perform better
- Model 1 On given Data - Observations:
    1. Model was trained for 19 epoch before early stopping was triggered. 
    2. Training and Validation accuracy increased gradually till 8 epochs but final accuracy was less than 60%.
    3. Model is overfitting as we see final Training Accuracy is higher than Validation Accuracy.
    4. Training Loss is also less than Validation loss which shows model is not able to predict well on validation data.
- Model 2 on Augumented Data - Observations:
    1. With Data Augmentation layer added to the model, there is an improvement in model and is no longer overfitting. 
    2. Accuracy for both Training and Validation is less than 55% but the validation accuracy is better than training accuracy.
    3. Losses are minimized (loss: 1.3701 & val_loss: 1.3879)
- Model 3 on given data and Augumented Data - Observations:
    1. Both training and validation accuracy has increased with combined data and model is not overfitting.
    2. Validation Loss is higher than training loss. 
    3. Final model paras
        Training Accuracy  : 83.00%,   Loss: 0.4472
        Validation Accuracy: 81.07%,   Loss: 0.6243
    
- Overfitting can be reduced by 
    1. Data Augumentation - Make sure you have enough samples and there is no class imbalance
    2. Dropout layers - Add the Dropout layers to get the smooth curve 
    3. Batch Normalization - This helps in reducing the quantities of the coeff and helps convergence.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- NumPy version: 1.25.2
- Pandas version: 2.0.3
- TensorFlow version: 2.17.0
- Augmentor version: 0.2.12

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- This project was part of Upgrad Assignment

## Contact
Created by [@rpjagtap2610] - feel free to contact me!


<!-- Optional -->
<!-- ## License --> 
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->