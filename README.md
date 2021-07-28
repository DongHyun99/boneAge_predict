# BoneAge-Predict  

<img src="./images/flowchart.png">  

입력층으로 X-Ray 사진 파일과 Gender를 받아서 뻐나이를 출력하는 네트워크  
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
      
