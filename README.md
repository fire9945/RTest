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


The entire system is designed as a two-stage fusion process:
1. Attentive pooling stage 
2. T60 mapping stage

The AWSSDR, reverberant environmental feature that reflects the imbalance in the influence of the reverberation, is produced through attentive pooling stage, and used for T60 estimation in T60 mapping stage.
The parameters of two sub networks: the weight estimation network and mapping network are simultaneously optimized to minimize the squared error between estimated T60 and ground truth T60.

## Implementation

### Training
#### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py
#### (\* = training set{1,2} number)

##### The trained model is stored in checkpoints/'model_name.dnn'

### Evaluation
#### python 1_testset*.py


## Experimental Results

### Settings
#### Training data
Similar to the datasets of the ACE challenge[ref1], training data were generated by convolving anechoic speech by the RIRs and adding noise according to each SNR. The SNR levels of the training data were set to 0, 10, and 20 dB, which were the same SNR levels of the DEV dataset of the ACE Challenge. 

Two sets of training data were built. <br/>
Common : RIRs (538 measured RIRs with T60 values between 0.1 and 1.5s from publicly available databases),  <br/>
<t/>         noises (6 types) <br/>
Set 1 : anechoic speech (TIMIT corpus), total 29,052 utt. <br/>
Set 2 : anechoic speech (TIMIT corpus + additional short utterances), total 58,104 utt. <br/>

All data samples were resampled to a sampling frequency of 16 kHz because a sampling frequency of the TIMIT corpus was 16 kHz.


