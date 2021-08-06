# BoneAge-Predict  

<img src="./result/result_image/Network Architecture.png">  

입력층으로 X-Ray 사진 파일과 Gender를 받아서 뻐나이를 예측하는 딥러닝 모델  
## Depemdemcies

* pytorch 1.9.0
* pandas
* glob
* open-cv
* matplotlib

## DataSet  

RSNA Pediatric Bone Age Challenge(2017)를 사용  
test set은 challenge라서 그런지 boneage에 대한 라벨링이 되어있지 않아서  
validation set을 2개로 나누어 측정했다.  

    # Train - 12611 images
    # Val   -  800 images
    # Test  -  625 images 

bone_data의 test, train, validation 폴더에 각각의 image 파일을 넣어줘야 한다.
      
## Accuracy

>MAE: 14.04613 (month)  

개선중..

## To-do's

- [ ] image Augmentation (denormalize..)  
- [ ] 하얀데이터와 검은데이터의 픽셀 평균을 구해서 두 데이터를 구분해보고 따로 학습(test, validation, train의 데이터를 통일하고 다시 나누는 작업 필요),  
test 할 때도 데이터의 픽셀 값에 따라 두 모델을 따로 사용해보기 (4~5개 정도로 나누고 Augmentation을 이용해서 데이터의 양을 늘릴것)  
- [x] 손 사진 색감을 전체적으로 통일해보기 (현재 40으로 통일해서 학습 중)  
- [x] Gender, image data shuffle -> 랜덤으로 넣으면 loss가 올라가므로 랜덤하지 않게 넣는 방법을 찾아볼 것  
- [ ] change Activation Function (Layer의 output을 보고 
    만약 음수 값이 나오지 않는 경우, ReLU를 사용할 필요가 X)
- [ ] change backbone Network: SE ResNext ->efficientNet V2 (classification으로 되어있어서 regression으로 바꿔야 함)

