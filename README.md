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

  
## Implementation

### Training
#### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py
#### (\* = training set{1,2} number)

##### The trained model is stored in checkpoints/'model_name.dnn'

  
### Evaluation
#### python 1_testset*.py


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
  
  [8]	M. Jeub, M. Schafer, and P. Vary, “A binaural room impulse response database for the evaluation of dereverberation algorithms,” in Proc. Intl. Conf. Digital Signal Process., Santorini, Greece, Jan. 2009, pp. 1–5.
  
  [9]	K. Kinoshita, M. Delcroix, T. Yoshioka, T. Nakatani, E. Habets, R. Haeb-Umbach, V. Leutnant, A. Sehr, W. Kellermann, R. Maas, S. Gannot, and B. Raj, “The reverb challenge. A common evaluation framework for dereverberation and recognition of reverberant speech,” in Proc. IEEE WASPAA, New Paltz, NY, USA, Oct. 2013, pp. 1–4.
  
  [10]	S. Nakamura, K. Hiyane, F. Asano, T. Nishiura, and T. Yamada, “Acoustical sound database in real environments for sound scene understanding and hands-free speech recognition,” in Proc. Intl. Conf. Language Resources and Evaluation, Athens, Greece, May 2000, pp. 965–968.
  
  [11]	D. T. Murphy and S. Shelley, “Opeanair: an interactive auralization web resource and database,” in Proc. 129th Audio Eng. Soc. Conv., San Francisco, California, USA, Nov. 2010.
  
  [12]	N. Parihar and J. Picone, “Aurora working group: DSR front end LVCSR evaluation au/384/02,” Inst. Signal and inf. Process., Mississippi State Univ., USA, Rep., 2002.
  
  [13]	J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, D. S. Pallett, and N. L. Dahlgren, “DARPA TIMIT acoustic-phonetic continuous speech corpus CD-ROM. NIST speech disc 1-1.1,” NASA STI/Recon, USA, Tech. Rep. NISTIR 4930, vol. 93, 1993.
  
  [14]	Surfing Technology Ltd., “ST-AEDS-20180100 1, Free ST American English Corpus,” Open Speech and Language Resources (OpenSLR), 2018. [Online]. Available: http://www.openslr.org/45/ 
  [15]	B. Kim, M. Lee, J. Lee, Y. Kim, and K. Hwang, “Query-by-example on-device keyword spotting,” in Proc. IEEE ASRU, Sentosa, Singapore, Dec. 2019, pp. 532–538.
  
  [16]	P. Warden, “Speech commands: a dataset for limited-vocabulary speech recognition,” 2018. [Online]. Available: https://arxiv.org/abs/1804.03209
  



  
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

  
### Analysis

#### FDSDD : Filterbank dependent SDD
We evaluated the T60 estimation performance combining the conventional SDR approaches with our mapping network in order to compare how much each of the two stages in our proposed method, namely attentive pooling stage and T60 mapping stage, contributes to the overall performance. 
- It is difficult to optimize the deep learning-based mapping network with the one-dimensional reverberant environmental features used in the SDD and SDDSA methods
- We constructed a reverberant environmental feature by obtaining the NSVs used in the SDD method for each filterbank and trained the mapping network using it as input to the T60 mapping stage.
- We refer to this method as filterbank dependent SDD (FDSDD).

![T60 estimation results by noise type and SNR level](https://user-images.githubusercontent.com/26379533/151282004-3ea03c5a-85f0-4269-8275-d32c27238fce.PNG)
where the results of SDDSA method are reported in [17]
  


- For high SNR, FDSDD method outperforms the SDDSA method (Unlike SDDSA, it does not consider noise)
  
  

  
  [17]	J. Eaton, N. D. Gaubitch, A. H. Moore, and P. A. Naylor, “Acoustic characterization of Environments (ACE) challenge results technical report,” Tech. Rep., Imperial College London, 2017. [Online]. Available: https://arxiv.org/abs/1606.03365
  
  
As shown in Fig. 4. we visualized AWSSDR and FDSDD using t-SNE [29] to observe whether these features are distinguished according to T60.    
<t-SNE 그림 추가>
