## Multi-branch Disruptive Information Transmission Network For Multi-lesion Semantic Segmentation

Paper Address : https://github.com/YF-W/DM-DICNet

Semantic segmentation technology has become a key force in advancing the medical system, as it can accurately identify and delineate lesions in medical images. The U-shape network has become the primary tool for semantic segmentation, but single-branch structures struggle with efficient information transmission, and neural networks often fail to capture critical information. Furthermore, simple bottlenecks are insufficient for extracting rich, deep-layer information. Lastly, most networks can only handle a limited number of lesion types and cannot be extended to multiple datasets.

## Paper:DM-DICNet(Multi-branch Disruptive Information Transmission Network For Multi-lesion Semantic Segmentation)

Authors  :  Yuefei Wanga, *, Yutong Zhanga, Yuxuan Wanb, Liangyan Zhaoa, Qinyu Zhaob, Guokai Chena

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

### Performance

$$
\text{IndicatorSuperiority}(Netx, Nety) = \frac{1}{n_D} \left( \sum_{i=1}^{n_D} \left( Netx_{D_i}^{ind} - Nety_{D_i}^{ind} \right) \right) \tag{7}
$$

| Method | Dice | IOU | LUNG | DRIVE | POLYP | BRAIN | TOOTH | SK | THYROID NODULE | LIVER | SYNAPSE |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepLab V1 | 22.07 | 22.79 | 4.65 | 51.53 | 35.2 | 29.34 | 13.32 | 26.3 | 12.26 | 11.97 | 17.28 |
| DeepLab V2 | 25.65 | 26.11 | 7.86 | 50.76 | 35.64 | 49.6 | 13.87 | 30.33 | 10.94 | 13.76 | 20.15 |
| DeepLab V3 | 25.41 | 24.92 | 3.78 | 45.79 | 40.14 | 58.43 | 13.59 | 32.57 | 11.96 | 7.93 | 12.3 |
| FCN | 15.22 | 15.81 | 0.9 | 39.57 | 36.73 | 12.17 | 9.91 | 19.75 | 8.25 | 6.07 | 6.27 |
| MSDANet | 9.46 | 10.41 | 0.97 | 15.09 | 23.12 | 10.03 | 1.82 | 23.63 | 3.27 | 2.25 | 9.24 |
| OAUNet | 9.31 | 10.58 | 1.81 | 6.43 | 29.41 | 11.71 | 3.16 | 21.51 | 5.59 | 2.84 | 7.06 |
| RefineNet | 32.21 | 30.55 | 3.15 | 33.69 | 33.98 | 32.32 | 58.43 | 14.8 | 19.99 | 7.34 | 78.73 |
| ResUNet | 15.48 | 16.65 | 2.68 | 3.97 | 43.25 | 24.09 | 2.11 | 35.24 | 21.28 | 5.74 | 6.19 |
| SegNet | 12.39 | 13.63 | 1.2 | 17.92 | 33.66 | 16.77 | 2.76 | 17.92 | 10.2 | 7.72 | 8.95 |
| SmaAt_UNet | 6.42 | 6.91 | 1.11 | 9.64 | 19.7 | 4.97 | 0.92 | 14.41 | 3.57 | 2.23 | 3.47 |
| UNet | 10.13 | 10.98 | 0.68 | 3.96 | 39.02 | 15.7 | 1.62 | 18.33 | 7.05 | 6.39 | 2.29 |
| UNet++ | 9.9 | 10.67 | 0.34 | 0.57 | 38.64 | 20.47 | 1.57 | 19.09 | 6.44 | 3.54 | 1.91 |
| UNet 3+ | 56.2 | 55.41 | 64.74 | 41.22 | 57.84 | 57.88 | 19.49 | 51.36 | 67.63 | 80.51 | 61.6 |
| UNet V2 | 12 | 12.89 | 0.76 | 31.37 | 27.25 | 6.37 | 2.76 | 9.32 | 13.49 | 8.63 | 12.05 |
| BRAU-Net++ | 8.16 | 9.15 | 0.72 | 11.07 | 27.9 | 6.31 | 0.41 | 13.51 | 6.54 | 4.43 | 6.97 |
| FTU_Net | 1.96 | 1.91 | 0.45 | 1.3 | 0.54 | 2.93 | 1.51 | 8.16 | 1.35 | 1.41 | -0.24 |
| HSH-UNet | 59.23 | 58.11 | 59.63 | 41.53 | 67.6 | 57.45 | 36.83 | 51.1 | 73.49 | 79.35 | 61.01 |
| MBP-SSNet | 10.05 | 11.31 | 0.99 | 9.38 | 23.47 | 16.06 | 1.97 | 21.16 | 8.22 | 4.21 | 10.69 |
| MBSNet | 8.05 | 9.07 | 3.05 | 9.03 | 21.36 | 9.55 | 2.09 | 21.75 | 4.27 | 2.9 | 3.03 |
| MTUNet | 49.14 | 45.11 | 55.66 | 1.41 | 73.5 | 69.49 | 76.63 | 30.37 | 79.39 | 35.52 | 2.15 |
| Subregion_Unet | 8.1 | 9.32 | 1.54 | 14.04 | 27.53 | 3.71 | 3.51 | 4.7 | 7.96 | 3.11 | 12.3 |
| SUNet | 27.68 | 27.35 | 2.89 | 47.65 | 60.6 | 18.93 | 7.72 | 24.47 | 51.61 | 11.53 | 22.27 |
| Swin_Unet | 25.81 | 25.61 | 2.17 | 43.58 | 61.86 | 20.55 | 4.14 | 21.98 | 45.6 | 12.15 | 19.35 |
| TransCeption | 12.59 | 13.07 | 1.26 | 31.74 | 36.62 | 5.81 | 1.31 | 11.15 | 10.87 | 8.73 | 7.98 |
| TransUNet | 5.34 | 6.59 | 0.99 | 5.75 | 11.67 | 1.35 | 0.01 | 6.09 | 6.18 | 7.86 | 13.8 |
| UNetFormer | 3.12 | 3.5 | 0.67 | 18.33 | 0.71 | -0.53 | 3.91 | 2.05 | 0.08 | 0.76 | 3.78 |


### Datasets:

1. The LUNG  dataset:https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
2. The POLYP  dataset:https://polyp.grand-challenge.org/CVCClinicDB/
3. The DRIVE dataset:https://drive.grand-challenge.org/
4. The SK  dataset:https://challenge.isic-archive.com/data/#2017
5. The BRIAN MRI  dataset:https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
6. The TOOTH  dataset:https://tianchi.aliyun.com/dataset/156596
7. The THYROID NODULE  dataset:https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st
8. The LIVER  dataset:https://www.kaggle.com/datasets/zxcv2022/digital-medical-images-for--download-resource
