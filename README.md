## Multi-branch Disruptive Information Transmission Network For Multi-lesion Semantic Segmentation

Paper Address:

Semantic segmentation technology has become a key force in advancing the medical system, as it can accurately identify and delineate lesions in medical images. The U-shape network has become the primary tool for semantic segmentation, but single-branch structures struggle with efficient information transmission, and neural networks often fail to capture critical information. Furthermore, simple bottlenecks are insufficient for extracting rich, deep-layer information. Lastly, most networks can only handle a limited number of lesion types and cannot be extended to multiple datasets.

## Paper:DM-DICNet(Multi-branch disruptive information transmission network for multi-lesion semantic segmentation)

Authors : Yufei Wang , Yutong Zhang , Li Zhang , Yuquan Xu , Yuxuan Wan , Zhixuan Chen ,  Liangyan Zhao , Qinyu Zhao , Guokai Chen , Ruixin Cao , YIxi Yang , Xi Yu

### 1. Architecture Overview

![image-20240722150751700](https://github.com/user-attachments/assets/d9931ac9-d3a3-4e52-a715-2f6c003d8423)


In our encoder-decoder structure, the Global Information Capture Branch (GICB) uses atrous convolution technology to capture global information, producing smooth edges while retaining core information to aid the network in understanding the relationship between lesions and surrounding tissues. The Core Information Capture Branch (CICB) enhances the central features of the GICB by processing cropped images and compensates for the loss of global information through information sharing, ensuring coherence between global and local features. The ResNet branch focuses on refining local information, utilizing residual structures to avoid gradient vanishing and explosion, and improving the ability to capture subtle structural changes, thereby significantly enhancing the quality of the segmentation results.

### 2. Our network baseline



![image-20240722144430738](https://github.com/user-attachments/assets/a85c7734-b0d0-4402-bd0d-72a33ca510fa)


We propose a baseline structure incorporating multiple encoders and decoders, utilizing a "three-encoder, two-decoder" strategy to effectively capture semantic information at different scales and complement each other, thereby reducing information loss. Architecturally, we employ a "directional transmission" method where the Global Information Capture Branch (GICB) and the Core Information Capture Branch (CICB) enhance network performance by mutually transmitting information. The ResNet branch, with its residual connections, effectively extracts multi-layer semantic features, significantly improving the segmentation accuracy and boundary feature extraction for lung lesions, liver, and brain MRI images.

### 3. Module 1: SDRE

![image-20240722150838013](https://github.com/user-attachments/assets/eadc15c0-628a-4ebd-8bf7-b103bb38f7a2)

To avoid the simple identity mapping that introduces shallow image noise into the deeper network, we designed a module within the skip connection section to enhance information processing and remove irrelevant information and noise. Additionally, we utilize Fast Fourier Convolution (FFC) with three kernels to transform the image from the spatial domain to the frequency domain, thereby expanding the receptive field and distinguishing core from secondary semantic information. We also split the image into edge and central regions, applying different operations to enhance segmentation performance and accuracy.

### 4. Module 2: SCHFE

![image-20240722150906588](https://github.com/user-attachments/assets/ad7b98de-6bce-4b84-b4ca-137cc43cdf2e)


In neural network architectures, the Bottleneck serves as a bridge between the encoder and decoder, concentrating highly abstract and diverse features. However, existing Bottlenecks are often shallow, limiting their ability to delve deeply into information. To address this, we propose the SCHFE module, which employs a "split and merge" strategy, decomposing the input image into two paths: one for clear and the other for blurred information. These paths process detail and overall information separately, and are merged by an Information Anti-Aliasing Branch (IAB) for deep extraction. The spiral contraction-based channel reduction mechanism within the module facilitates seamless integration of multi-scale features, significantly enhancing the recognition accuracy of hidden lesions (e.g., polyps) and complex images (e.g., retina vessels), reducing the risk of overfitting, and improving the model’s generalization performance and stability.

### Datasets:

1. The LUNG  dataset:https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
2. The POLYP  dataset:https://polyp.grand-challenge.org/CVCClinicDB/
3. The DRIVE dataset:https://drive.grand-challenge.org/
4. The SK  dataset:https://challenge.isic-archive.com/data/#2017
5. The BRIAN MRI  dataset:https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
6. The TOOTH  dataset:https://tianchi.aliyun.com/dataset/156596
7. The THYROID NODULE  dataset:https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st
8. The LIVER  dataset:https://www.kaggle.com/datasets/zxcv2022/digital-medical-images-for--download-resource
