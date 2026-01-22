# Using-CNNs-for-Breast-Cancer-Histology-Detection
(NOTE: This project was my Year 12 Semester 1 2025 project. This won 2nd Place in the ACT in Young ICT Explorers (YICTE) Australia competition in 2025. I also presented this project at PyCon AU 2025 in the Student Showcase -> link to YouTube video: https://www.youtube.com/watch?v=mqg93zv1S-E&t=5659s)

## Project summary:
This is a computer vision project that implements a Convolutional Neural Network (CNN) to classify images of breast tissue, detecting whether or not a tumour is present. This is an area of deep learning that specifies in pattern recognition, especially useful for image detection. 

This means that tumours can be detected within images, without the underlying scientific knowledge of how to visually identify a tumour. With enough training, and exposure to a variety of images of different classifications, the model learns to detect visual patterns and classify an image as Malignant (cancerous) or Benign (non-cancerous).  

The project implements a CNN that is a hybrid of two models. A pretrained ResNet-18 model was implemented as the feature extractor. For the classification layer, a very basic classifier-head inspired by TinyVGG was implemented - containing a ReLu function for the activations and a Linear layer. The CNN optimises the parameters of the hidden layers to attain more accurate classification each epoch, using backpropagation, which is based on gradient descent. 

## About the dataset:
This project classifies images from the BreakHis dataset, which contains 9109 microscopic breast tissue images collected from 82 patients, and has 2480 benign and 5429 malignant samples, with different subcategories of benign and malignant, and magnifications. For simplicity, I chose the subcategories “Ductal carcinoma” in malignant images, and “fibroadenoma” in benign images, both at 40x magnification. An 80-20 split gave 893 training and 224 testing images. 

**Link to dataset:** https://www.kaggle.com/datasets/ambarish/breakhis

## Problem statement:
Breast Cancer is the most commonly diagnosed cancer amongst women in Australia. According to the National Breast Cancer Foundation, approximately 57 women are diagnosed with Breast Cancer every day, meaning over 20000 women are diagnosed with this cancer every year. 1 in 7 women are diagnosed with Breast Cancer in their lifetime. Currently, approximately one woman under the age of 40 is expected to die each week from breast cancer. In the last 10 years, breast cancer diagnosis has increased by 33%. 

AI helps pathologists by classifying tissue images automatically. Manually analysing histology slides is time-consuming and prone to human error. Therefore, automating this process using a computer vision model mitigates these drawbacks, and allows for early detection. Early detection is the key to saving lives as the earlier the cancer diagnosis, the decreased risk of the cancer spreading, which in turn, increases survival rates.

## Design features:
This project classifies images from the BreakHis dataset, which contains 9109 microscopic breast tissue images collected from 82 patients, and has 2480 benign and 5429 malignant samples, with different subcategories of benign and malignant, and magnifications. For simplicity, I chose the subcategories “Ductal carcinoma” in malignant images, and “fibroadenoma” in benign images, both at 40x magnification. An 80-20 split gave 893 training and 224 testing images. 


The project implements a CNN. For my CNN, I have integrated two known models, called ResNet-18, and TinyVGG. 

#### ResNet-18 for feature extraction

ResNet-18 is a pre-trained model that is especially good for histology dataset classification tasks – requiring deeper networks. It comprises of 18 convolutional layers, that utilise residual blocks to address the vanishing gradient problem * in deeper networks.

This is a problem that arises when the gradients become very small as they are propagated through layers - often causing early layers to receive near-zero gradient updates. (i.e., When the gradients of multiple layers are multiplied together, the result is often incredibly close to zero - this is done when the chain rule is applied during backpropagation).

In a normal CNN, the output of the previous layer is taken, and is replaced with something completely new in the next layer - each layer transforms the results, meaning you lose the original input by the time you get to the end. In ResNet, you keep the original input, and add it back later after going through some layers.


<img width="800" height="250" alt="image" src="https://github.com/user-attachments/assets/b6bd35c0-903b-4f13-a3e9-e068f844464c" />

This creates a shortcut called a skip connection - and these are the arrows that show this. So, these arrows mean that the input is added back after a couple of convolutions.


These skip connections give the model a kind of memory - avoiding information loss - by providing the network a reminder of where things started, so that deeper layers can decide what to keep and what to define. A residual block is the chunk of layers that add the input back in. Essentially, ResNet learns how to slightly tweak the input to produce the output, without learning all the patterns from scratch - which is way more efficient, and faster to train - meaning that you can train hundreds of layers and still improve performance, without worrying about stagnant progress, due to suffering from vanishing gradients.

The residual blocks ensure that important information is preserved instead of accidentally forgotten or distorted by transformations, and also provide a safety net - if the network tries to make a change that hurts performance, it can “fall back” to the input and try again.

I have used ResNet-18 as my feature extractor - to identify the patterns in the image. I’ve implemented all layers up till the last hidden, Fully Connected (FC) layer.

#### Tiny-VGG Inspired Classifier
The final, fully connected layer is the classifier head. I have constructed the classifier head design similar to TinyVGG’s FC layer. This is a dense layer in which the data is flattened, features are mapped into a smaller space (the same way this is done throughout the previous layers). the ReLu activation function is used between layers, and a class prediction is outputted at the end.

<img width="480" height="200" alt="image" align="left" src="https://github.com/user-attachments/assets/ee6dddee-ee55-48f8-8a0d-7b836a67b256" />
<img width="490" height="200" alt="image" align="right" src="https://github.com/user-attachments/assets/3c141ec6-3806-47e1-b8bd-cece1e1f5c0a" />

Essentially, I replaced the last FC from ResNet-18 with a classifier that is personally developed, and inspired by TinyVGG. My classifier is simpler than ResNet’s original FC layer.

ResNet-18 is originally pretrained on the ImageNet dataset, which has a 1000 output classes. However, we are not aiming for multi-class classification in this project, so for binary classification, it is optimal to utilise a personally developed classifier head - especially to prevent overfitting.


### <ins>Final hybrid CNN</ins>– combining ResNet feature extractor and custom classifier head
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/1e39d5b1-72f0-47ec-ab75-b3cf213bdd4c" />

Therefore, this is how the CNN has integrated the two models. ResNet-18’s hidden layers have been utilized for the feature extraction, up till the last Fully Connected (FC) layer (meaning ResNet-18’s last layer has not been used). The final classification layer of ResNet-18 has been replaced with the TinyVGG inspired classifier head. 

## Challenges designing and building the project
### 1. Vanishing gradient problem **
   Explanation provided in the outline of the ResNet-18 architecture above.
### 2. Gradient descent stagnation problem (potential issue in future)
   One of the most well-known issues with gradient descent in backpropagation is something I might only face when I build ResNet entirely from scratch, instead of utilizing the pre-trained model.

   The process of converging towards a local minimum in gradient descent is often ineffective, when a relatively flat area in the loss function curve is reached. Gradient descent often gets stuck at a local minimum, instead of continuing to approach a lower global minimum instead.

   This means the algorithm gets trapped in a plateau or local minimum where the gradient is nearly zero, leading to very slow progress or no further improvement in performance. In the future, if I get a chance to write ResNet-18 from scratch, I hope to learn how to solve such an issue.
### 3. Typo – when selecting layers of ResNet-18
   As stated before, for the Hybrid CNN, I replaced only the last layer of ResNet-18 with a custom classifier head inspired by TinyVGG, which means I had to select all the layers up till the last layer.

   I made a small typo here. Instead of writing what I should’ve written in the green, I had written what was in the red, without the colon that I should have included.

   This meant that instead of selecting all the layers from ResNet except for the last prediction layer, I was selecting only the last layer, which meant that I had no feature extraction being performed. Without any feature extraction, the dataset was being classified using the final FC classification layer from ResNet-18, and then again with the TinyVGG inspired classifier head.

   <img width="900" height="350" alt="image" src="https://github.com/user-attachments/assets/def57f34-a513-4930-8cf5-7bad77a9f2fd" />

   This dropped my accuracy down to about 20% on the training data, proving the importance of having adequate feature extraction. Once I fixed this typo the accuracy went up to 96% - meaning having a high performing feature extractor is crucial to achieve high classification accuracy. 

## Photos
#### <ins>Tiny-VGG Inspired Classifier</ins>
<img width="980" height="250" alt="image" src="https://github.com/user-attachments/assets/cf8611ac-9c0d-4e76-b87d-3514818e1e1a" />




