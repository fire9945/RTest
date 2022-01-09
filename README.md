# Attentive Pooling-Based Weighted Sum of Spectral Decay Rates for Blind Estimation of Reverberation Time 


![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=yellow)
![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=black)


![header](https://capsule-render.vercel.app/api?type=Rect&color=auto&height=200&section=footer&text=Blind%20T60%20Estimation&fontSize=70&textBg=true)


## About
This repo contains the implementation and experimetal results of a method for blind revereberation time (T60) estimation using Attentive pooling-based Weighted Sum of Spectral Decay Rates (AWSSDR). 

## Implementation
* = training set number
### Training
#### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py

##### The trained model is stored in checkpoints/'model_name.dnn'

### Evaluation
#### python 1_testset*.py







Spectral Decay Rates (SDRs) which represent the degree to which the energy of a speech signal decays over time in the time–frequency domain 
The AWSSDR is a environmental feature for blind estimation of reverberation time. An entire system of the T60 estimation method using AWSSDR is as follows:

![전체흐름도 수정](https://user-images.githubusercontent.com/26379533/148699432-aaa7a6f6-594f-40db-94db-6ffc41126f34.PNG)
