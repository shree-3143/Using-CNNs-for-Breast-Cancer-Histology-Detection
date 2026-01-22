# Using-CNNs-for-Breast-Cancer-Histology-Detection
(NOTE: This project was my Year 12 Semester 1 2025 project. This won 2nd Place in the ACT in Young ICT Explorers (YICTE) Australia competition in 2025. I also presented this project at PyCon AU 2025 in the Student Showcase -> link to YouTube video: https://www.youtube.com/watch?v=mqg93zv1S-E&t=5659s)

### Project summary:
This is a computer vision project that implements a Convolutional Neural Network (CNN) to classify images of breast tissue, detecting whether or not a tumour is present. This is an area of deep learning that specifies in pattern recognition, especially useful for image detection. 

This means that tumours can be detected within images, without the underlying scientific knowledge of how to visually identify a tumour. With enough training, and exposure to a variety of images of different classifications, the model learns to detect visual patterns and classify an image as Malignant (cancerous) or Benign (non-cancerous).  

The project implements a CNN that is a hybrid of two models. A pretrained ResNet-18 model was implemented as the feature extractor. For the classification layer, a very basic classifier-head inspired by TinyVGG was implemented - containing a ReLu function for the activations and a Linear layer. The CNN optimises the parameters of the hidden layers to attain more accurate classification each epoch, using backpropagation, which is based on gradient descent. 

### About the dataset:
This project classifies images from the BreakHis dataset, which contains 9109 microscopic breast tissue images collected from 82 patients, and has 2480 benign and 5429 malignant samples, with different subcategories of benign and malignant, and magnifications. For simplicity, I chose the subcategories “Ductal carcinoma” in malignant images, and “fibroadenoma” in benign images, both at 40x magnification. An 80-20 split gave 893 training and 224 testing images. 

**Link to dataset:** https://www.kaggle.com/datasets/ambarish/breakhis

### Problem statement:
Breast Cancer is the most commonly diagnosed cancer amongst women in Australia. According to the National Breast Cancer Foundation, approximately 57 women are diagnosed with Breast Cancer every day, meaning over 20000 women are diagnosed with this cancer every year. 1 in 7 women are diagnosed with Breast Cancer in their lifetime. Currently, approximately one woman under the age of 40 is expected to die each week from breast cancer. In the last 10 years, breast cancer diagnosis has increased by 33%. 

AI helps pathologists by classifying tissue images automatically. Manually analysing histology slides is time-consuming and prone to human error. Therefore, automating this process using a computer vision model mitigates these drawbacks, and allows for early detection. Early detection is the key to saving lives as the earlier the cancer diagnosis, the decreased risk of the cancer spreading, which in turn, increases survival rates.

### Design features:
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

<img width="500" height="100" alt="image" align="left" src="https://github.com/user-attachments/assets/ee6dddee-ee55-48f8-8a0d-7b836a67b256" />
<img width="500" height="100" alt="image" align="right" src="https://github.com/user-attachments/assets/3c141ec6-3806-47e1-b8bd-cece1e1f5c0a" />

