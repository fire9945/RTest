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
<p align="center">Fig. 1. Illustration of proposed T60 estimation method using the AWSSDR.<p align="center">


The entire system is designed as a two-stage fusion process:
1. Attentive pooling stage 
2. T60 mapping stage

The AWSSDR, reverberant environmental feature that reflects the imbalance in the influence of the reverberation, is produced through attentive pooling stage, and used for T60 estimation in T60 mapping stage.
The parameters of two sub networks: the weight estimation network and mapping network are simultaneously optimized to minimize the squared error between estimated T60 and ground truth T60.

## Experimental Results

### Settings
#### Training Data
Similar to the datasets of the ACE challenge[ref1], training data were generated by convolving anechoic speech by the RIRs and adding noise according to each SNR. The SNR levels of the training data were set to 0, 10, and 20 dB, which were the same SNR levels of the DEV dataset of the ACE Challenge. 

Two sets of training data were generated. <br/>
Common : RIRs (538 measured RIRs with T60 values between 0.1 and 1.5s from publicly available databases), noises (6 types) <br/>
Set 1: anechoic speech (TIMIT corpus), total 29,052 utt. <br/>
Set 2: anechoic speech (TIMIT corpus + additional short utterances), total 58,104 utt. <br/>
**_Most algorithms submitted to the ACE Challenge performed better with longer utterances. Considering this, we tried to improve the performance of T60 estimation for short utterances by generating additional training data (set 2) including short utterances._**

All data samples were resampled to a sampling frequency of 16 kHz.

#### Model Training
The pytorch deep learning framework was used for implementing proposed method. When obtaining a batch of data by sequence level, the lengths of the sequences were fitted to the longest sequence in the batch using padding to batch various lengths of sequences. If a large batch is used, the difference in the length of the sequences included in the batch becomes large. Because of this, short sequences are excessively padded, which lowers the stability of the training, as shown in Fig. 2(a). Therefore, we batched the training sequences according to their length order and used a small batch. To achieve the same effect as using a large batch we updated the model at once after calculating the gradients of multiple batches. Fig. 2(b) shows that this method improves the training stability. The batch size was set to 16, and the model was updated every 16 batches.


<center><img src="https://user-images.githubusercontent.com/26379533/148746078-0feeb270-f866-47cc-ba34-9bc585dcb6fa.png" width="800" height="200"/></center>
<p align="center">(a)<p align="center">
  
<center><img src="https://user-images.githubusercontent.com/26379533/148746576-e2dccf24-32c2-486d-8685-9c31fea76feb.png" width="800" height="200"/></center>  
<p align="center">(b)<p align="center">
  
<p align="center">Fig. 2. Loss tracking of training and validation data. (a) Training with the large batch. (b) Training with the small multi-batch.<p align="center">


### Results
#### Performance comparison of blind T60 estimation
Evaluation data: ACE Challenge EVAL set

  
\begin{table}[t!]
\caption{Performance Comparison of Blind T$_{60}$ Estimation}
\label{experiment results}
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{16pt}
\centering
\begin{tabular}{cccc}
\hline\hline
Algorithm &	Bias & MSE	& ${\rho}$ \\
\hline
QA Reverb [4] &	-0.0680 &	0.0648 &	0.778 \\
\hline
SDDSA [6] &	-0.0423 &	0.0803 &	0.600\\
\hline
MLP [7] &	-0.0967 &	0.1040 &	0.480 \\
\hline
CNN [8] &	0.0304 &	0.0384 &	0.836 \\
\hline
CNN+AIRA [9] &	-0.0264 &	0.0261 &	0.920\\
\hline
CRNN [10] &	-0.0488 & 	0.0206 &	0.917 \\
\hline
AWSSDR (set 1) &	-0.0091 &	0.0166 &	0.936\\
\hline
AWSSDR (set 2) &	0.0268 &	0.0131 &	0.953 \\
\hline\hline 
\end{tabular}
\end{table}


  Fullband T60 estimation results by parameter
  FB T60 estimation error in all noises and all SNRs for a) female talkers and b) male talkers
  Single channel FB T60 estimation error in all noises and all SNRs for a) T60 †0.43 s b) 0.43 §T60 † 0.75 s and c) T60 • 0.75 s. Observe
that ⇢ † 0 for all except algorithm D
  FB T60 estimation error in all noises at a), 18 dB SNR, b), 12 dB SNR, and c) ´1dB SNR
  FB T60 estimation error in all noises and all SNRs for a) utterance length †5 s b) utterance length † 15 s and c) utterance length • 15 s
  
T60 estimation algorithm performance in all noises for all SNRs
  
Fullband T60 estimation results by noise type
  Ambient noise
  Fullband T60 estimation error in ambient noise for all SNRs
  T60 estimation algorithm performance in ambient noise for all SNRs



## Implementation

### Training
#### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py
#### (\* = training set{1,2} number)

##### The trained model is stored in checkpoints/'model_name.dnn'

### Evaluation
#### python 1_testset*.py


