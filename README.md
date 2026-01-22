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

<img width="400" height="100" alt="image" src="https://github.com/user-attachments/assets/cbf7d3b3-816b-4286-8ab6-0148984b4127" />


