# Traffic-Sign-Classifier-CNN
A neural network for classifying traffic signs. The neural networks is a CNN (Convolutional Neural Network). 
## Introduction:
Road safety is a critical issue worldwide, with millions of people being killed or injured in road accidents each year. A primary factor contributing to these accidents is the negligence of drivers in obeying traffic regulations, especially in recognizing and adhering to traffic signs. Traffic signs are critical components of the road transportation system as they provide essential guidance to drivers and other road users. Traffic sign recognition is an important area in computer vision and accurate recognition of traffic signs is important for the safety of drivers and pedestrians. Traffic sign classification can be used by an agent (self-driving cars) to identify and interpret traffic signs to make informed driving decisions in a partially observable environment. An agent that is equipped with traffic sign classifier can help improve road safety significantly. Also, it can help improve pedestrian safety by detecting pedestrian crossing signs and school zones. In this project, we plan to implement a CNN model to classify traffic signs.

## Objectives:
The main objective of this project is to implement a CNN model that can accurately classify traffic signs so that it can be deployed in self-driving cars. Specifically, we aim to:
1. Implement, train, and test different CNN architectures for image classification. We will implement and train a model using convolutional neural network (CNN) architecture.
2. Select image size (28 ́28, 64 ́64, or 128 ́128) based on performance measures.
3. Pre-process data before feeding into CNN model. For e.g., data augmentation
4.  Evaluate the models on test set based on different performance measures.
5.  Fine tune model hyper-parameters, architecture to achieve better performance on test dataset.
6.  
## Dataset Resources:
For the purpose of this project, we will use the traffic signs dataset given on the Kaggle website at https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification, which contains 58 classes and most classes have around 120 images. The dataset contains classes of traffic signs, including speed limits, stop signs, U-turn, warning signs, crossing signs and yield signs. The dataset has more than 6000 images including the test set.

## Prior work and Novelty:
A few people worked on this project using pre-defined CNN models with around 90% accuracy. But we will implement our own CNN model and architecture. We will try different techniques of pre-processing and data augmentation. We will try to add some noise and transformations in the dataset to simulate real world situation so in case if the images are not clear, the model still be able to classify them. We will evaluate model with different performance measures like f1-score, confusion matrix, precision and not just accuracy.
 
## Methods:
We will use Convolutional Neural Network which is very popular method for image classification in Artificial Intelligence. We will use Python, Keras, and TensorFlow libraries for Implementing, training, and evaluating the CNN model. The following steps will be followed:
1. Preprocessing: We will preprocess the images by normalizing the pixel values. We will also perform data augmentation to increase the size of the dataset and reduce overfitting if needed. Augmentation techniques such as rotation, scaling, and flipping will be used to increase the size and diversity of the dataset.
2. Model Selection: We will create, train, and test different CNN architectures for image classification. The model will consist of several convolutional layers, followed by pooling layers and fully connected layers. We will also experiment with different hyperparameters such as learning rate, batch size, and dropout rate.
3.  Model Evaluation: We will evaluate the performance of the models based on their accuracy and f1-score. We will also use confusion matrices to analyze the models' performance on different traffic sign classes. The model will be trained on 60% of the whole dataset, 20% will be used for validation, and then its performance will be evaluated on test dataset which is remaining 20% of whole dataset.

## Conclusion:
In this project, we aim to develop a CNN model for traffic signs classification. This project will provide valuable insights into the application of CNN models for image classification and their potential use in real-world applications, specifically in self-driving cars.
