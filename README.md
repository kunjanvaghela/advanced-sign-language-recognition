# Advancing Sign Language Recognition: A Comprehensive Study on Machine Learning Models for American and Indian Sign Languages

## Overview
Sign Language Recognition (SLR) is crucial for facilitating communication for the deaf and hard-of-hearing community. This project explores the use of machine learning to interpret American Sign Language (ASL) and Indian Sign Language (ISL). We performed a comparative analysis of various machine learning models including K-Nearest Neighbor (KNN), Support Vector Machine (SVM), DeepASLR, and Transfer Learning with pre-trained models such as VGG16 and InceptionV3. The study also investigated the impact of dataset augmentation techniques on model performance.

## Dataset Used
### ASL - American Sign Language
- **Dataset Size**: 27,455 training samples, 7,172 test samples
- **Image Dimensions**: 28x28 pixels, grayscale
- **Classes**: A to Z (excluding J and Z)

### ISL - Indian Sign Language
- **Dataset Size**: 13,689 training samples, 1,502 validation samples, 1,486 test samples
- **Image Dimensions**: 400x400 pixels, color
- **Classes**: Numerical values (0-9) and alphabets (A-Z)

## Methodology
### K-Nearest Neighbor (KNN)
- **Preprocessing**: Resizing, flattening, and min-max scaling of images
- **Model**: KNN classifier with Euclidean distance and varying K values
- **Evaluation**: Accuracy, precision, recall, and F1 score across skin tones

### Support Vector Machine (SVM)
- **Preprocessing**: Similar to KNN
- **Model**: SVM with linear and non-linear kernels
- **Hyperparameter Tuning**: GridSearchCV
- **Evaluation**: Accuracy, precision, recall, and F1 score across skin tones

### DeepASLR
- **Architecture**: Convolutional Neural Network (CNN)
- **Layers**: Convolution, pooling, flattening, fully connected layers
- **Implementation**: Keras, Tensorflow, OpenCV
- **Performance**: Achieved an average accuracy of approximately 99.42%

### Transfer Learning
- **Pre-trained Models**: VGG16 and InceptionV3
- **Approach**: Fine-tuning on ASL and ISL datasets
- **Advantages**: Saves training time, improves performance with smaller datasets

## Results
### KNN
- **ASL Accuracy**: Up to 99.64%
- **ISL Accuracy**: Up to 98.41%
- **Optimal K**: 7 for balanced performance
- **Performance Variation**: Higher misclassification in similar gestures (e.g., U, V, W for ASL)

### SVM
- **ASL Accuracy**: 99.79% (Linear Kernel), 97.34% (RBF Kernel)
- **ISL Accuracy**: 96.67% (Linear Kernel), 98.06% (RBF Kernel)
- **Performance**: Consistent high precision and recall

### DeepASLR
- **ASL Accuracy**: 85% after 50 epochs
- **ISL Accuracy**: Exceeds 98% after 25 epochs
- **Performance**: Higher for ISL due to double-handed gestures

### Transfer Learning
#### VGG16
- **ASL Accuracy**: 99.9%
- **ISL Accuracy**: Slightly lower than InceptionV3
#### InceptionV3
- **ASL Accuracy**: 99.7% (precision, recall, F1-score)
- **ISL Accuracy**: 98.54%

## Conclusion
Our study highlights the effectiveness of machine learning models in recognizing ASL and ISL gestures, emphasizing the benefits of dataset augmentation and transfer learning. The DeepASLR model showed exceptional performance, especially for ISL, and transfer learning with InceptionV3 provided superior results over VGG16.

## Contribution
My contribution to this group project was the implementation, training, and evaluation of the DeepASLR CNN model.

## References
1. [SIGN Dataset](https://example.com/sign-dataset)
2. [VGG16 Model](https://example.com/vgg16)
3. [InceptionV3 Model](https://example.com/inceptionv3)

For more details, please refer to the [project report]("Advancing Sign Language Recognition.pdf") <a name="Advancing Sign Language Recognition.pdf"></a>.
