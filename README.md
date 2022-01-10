# Attentive Pooling-Based Weighted Sum of Spectral Decay Rates for Blind Estimation of Reverberation Time 


![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=yellow)
![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=black)


![header](https://capsule-render.vercel.app/api?type=Rect&color=auto&height=200&section=footer&text=Blind%20T60%20Estimation&fontSize=70&textBg=true)


## About
This repo contains the implementation and experimetal results of a method for blind revereberation time (T60) estimation using Attentive pooling-based Weighted Sum of Spectral Decay Rates (AWSSDR). An entire system of the T60 estimation method using AWSSDR is as follows:

![전체흐름도 수정](https://user-images.githubusercontent.com/26379533/148712728-d47d4e74-f8b2-4966-a67f-a5f406ae63bf.PNG)



The entire system includes two sub networks:
1. Weight estimation network 
2. T60 mapping network.




## Implementation

### Training
#### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py
#### (\* = training set{1,2} number)

##### The trained model is stored in checkpoints/'model_name.dnn'

### Evaluation
#### python 1_testset*.py


