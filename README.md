# Attentive Pooling-Based Weighted Sum of Spectral Decay Rates for Blind Estimation of Reverberation Time 


![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=yellow)
![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=black)


![header](https://capsule-render.vercel.app/api?type=Rect&color=auto&height=200&section=footer&text=Blind%20T60%20Estimation&fontSize=70&textBg=true)


## About
This repo contains the implementation and experimetal results of a method for blind revereberation time (T60) estimation using an Attentive pooling-based Weighted Sum of Spectral Decay Rates (AWSSDR). <br/>
Spectral Decay Rates (SDRs), which represent the degree to which the energy of a speech signal decays over time in the time–frequency domain, are related to physical properties of reverberation. Conventional blind T60 esimation methods using SDRs select the SDRs that better represent the characteristics of reverberation, but they have limitations in that they employ a hard decision mechanism whether to select an SDR or not, and consider the selected SDRs to be equally important.  <br/> 
As the SDRs rely not only on reverberation, but also on speech context, noise, and speaker, not all SDRs retain equal amount of information about reverberation. 
Therefore, we introduce a soft decision mechanism that assigns a weight to each SDR according to the importance of the information about reverberation contained in each SDR, and estimate the weights by applying an attention mechanism in deep learning approaches. That's the AWSSDR! <br/>
An entire system of the T60 estimation method using AWSSDR is as follows:

![전체흐름도 수정](https://user-images.githubusercontent.com/26379533/148723147-5a081897-b4a9-445b-bb01-f650f7240269.PNG)
**<p align="center">Fig. 1. Illustration of proposed T60 estimation method using the AWSSDR.<p align="center">**


The entire system is designed as a two-stage fusion process:
1. Attentive pooling stage 
2. T60 mapping stage

The AWSSDR, reverberant environmental feature that reflects the imbalance in the influence of the reverberation, is produced through attentive pooling stage, and used for T60 estimation in T60 mapping stage.
The parameters of two sub networks: the weight estimation network and mapping network are simultaneously optimized to minimize the squared error between estimated T60 and ground truth T60.

## Experimental Results
  
### Performance comparison  

**Evaluation data: EVAL dataset of ACE Challenge [1]**
  
  ![performance table](https://user-images.githubusercontent.com/26379533/151280081-e1728498-23a5-4d4c-86f6-3c7fe843b05e.PNG)

- Table I shows a comparison of the performance of the proposed method and those of the six previously published methods. 
- The bias, MSE, and ρ were the evaluation criteria used in the ACE Challenge, and they represent the mean of the estimation error, mean squared error, and Pearson correlation coefficient between the estimated T60 and the ground truth T60, respectively. 
- As can be seen, the performance of our blind T60 estimation method outperforms previously published state-of-the-art methods for all evaluation criteria.
  
  [1]	J. Eaton, N. D. Gaubitch, A. H. Moore, and P. A. Naylor, “Estimation of room acoustic parameters: The ACE Challenge,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 24, no. 10, pp. 1681–1693, Oct. 2016.
  
  [2] T. de M. Prego, A. A. de Lima, R. Zambrano-López, and S. L. Netto, “Blind estimators for reverberation time and direct-to-reverberant energy ratio using subband speech decomposition,” in Proc. IEEE Workshop Appl. Signal Process. Audio Acoust., New Paltz, NY, USA, Oct. 2015, pp. 1–5.
  
  [3]	J. Eaton, N. D. Gaubitch, and P. A. Naylor, “Noise-robust reverberation time estimation using spectral decay distributions with reduced computational cost,” in Proc. IEEE Intl. Conf. Acoust., Speech, Signal Process., Vancouver, BC, Canada, May 2013, pp. 161–165.

  [4]	F. Xiong, S. Goetze, and B. T. Meyer, “Joint estimation of reverberation time and direct-to-reverberation ratio from speech using auditory-inspired features,” in Proc. ACE Challenge Workshop, Satell. Event IEEE-WASPAA, New Paltz, NY, USA, Oct. 2015, pp. 1–5.
  
  [5]	H. Gamper and I. J. Tashev, “Blind reverberation time estimation using a convolutional neural network,” in Proc. Intl. Workshop Acoust. Signal Enhancement, Tokyo, Japan, Sep. 2018, pp. 136–140.
  
  [6]	N. J. Bryan, “Impulse response data augmentation and deep neural networks for blind room acoustic parameter estimation,” in Proc. IEEE Intl. Conf. Acoust., Speech Signal Process., Barcelona, Spain, May 2020, pp. 5000–5004.
  
  [7]	S. Deng, W. Mack, and E. A. P. Habets, “Online blind reverberation time estimation using CRNNs,” in Proc. Interspeech, Shanghai, China, Oct. 2020, pp. 5061–5065.

### ACE Corpus and datasets

![ACE dataset descript](https://user-images.githubusercontent.com/26379533/151281226-d9468ac9-fcec-4628-a2e5-dd7f1b6944a1.PNG)

- The ACE corpus is a database consisting of anechoic speech utterances, RIRs, and noise that provides realistic noisy reverberant speech utterances for the development and evaluation of acoustic parameter estimation algorithms. 
- The ACE Challenge provides a development (DEV) dataset and evaluation (EVAL) dataset constructed from the ACE Corpus. Table I presents the speakers, utterances, RIRs, noises, and SNRs that comprise the DEV and EVAL datasets.

### Settings

#### Training Data

Two sets of training data were generated. <br/>
- Common : RIRs (538 measured RIRs with T60 values between 0.1 and 1.5s from publicly available databases{Aachen [8], the REVERB Challenge [9], RWCP [10], Openair [11]}), 
  noises (6 types noises from Aurora-4 task [12]) <br/>
  
- Set 1: anechoic speech (TIMIT corpus [13]), total 29,052 utt. <br/>
**_In [5], the specifications for noise types and SNR levels of DEV dataset are followed for generating training samples. Our training dataset (set 1) was configured similarly to the CNN method in terms of the range of T60, the anechoic speech, and the SNR level._**
- Set 2: anechoic speech (TIMIT corpus + additional short utterances{ST-AED[14], Qualcomm [15], Google speech command [16]}), total 58,104 utt. <br/>
**_Most algorithms submitted to the ACE Challenge performed better with longer utterances. Considering this, we tried to improve the performance of T60 estimation for short utterances by generating additional training data (set 2) including short utterances._**

- All data samples were resampled to a sampling frequency of 16 kHz.
  
Similar to the datasets of the ACE challenge[1], training data were generated by convolving anechoic speech by the RIRs and adding noise according to each SNR. The SNR levels of the training data were set to 0, 10, and 20 dB, which were the same SNR levels of the DEV dataset of the ACE Challenge and CNN method. 

#### Model Training
The pytorch deep learning framework was used for implementing proposed method. 
- When obtaining a batch of data by sequence level, the lengths of the sequences were fitted to the longest sequence in the batch using padding to batch various lengths of sequences. 
- If a large batch is used, the difference in the length of the sequences included in the batch becomes large. Because of this, short sequences are excessively padded, which lowers the stability of the training, as shown in Fig. 2(a). Therefore, we batched the training sequences according to their length order and used a small batch. 
- To achieve the same effect as using a large batch we updated the model at once after calculating the gradients of multiple batches. Fig. 2(b) shows that this method improves the training stability. The batch size was set to 16, and the model was updated every 16 batches.

<center><img src="https://user-images.githubusercontent.com/26379533/148746078-0feeb270-f866-47cc-ba34-9bc585dcb6fa.png" width="800" height="200"/></center>
<p align="center">(a)<p align="center">
  
<center><img src="https://user-images.githubusercontent.com/26379533/148746576-e2dccf24-32c2-486d-8685-9c31fea76feb.png" width="800" height="200"/></center>  
<p align="center">(b)<p align="center">
  
**<p align="center">Fig. 2. Loss tracking of training and validation data. (a) Training with the large batch. (b) Training with the small multi-batch.<p align="center">**



  
  
  
  
As shown in Fig. 4. we visualized AWSSDR and FDSDD using t-SNE [29] to observe whether these features are distinguished according to T60.    
<t-SNE 그림 추가>

## Implementation

### Training
#### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py
#### (\* = training set{1,2} number)

##### The trained model is stored in checkpoints/'model_name.dnn'

### Evaluation
#### python 1_testset*.py


