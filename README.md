# BoneAge-Predict  

## 1. Bone-Age  

출처: https://github.com/kneehit/Bone-Age

<img src="Bone-Age/images/flowchart.png">  

입력층으로 X-Ray 사진 파일과 Gender를 받아서 뻐나이를 출력하는 네트워크  

### dataSet  
    RSNA Pediatric Bone Age Challenge를 사용  
    train set만으로 train, validation, test set을 모두 사용한다.  
    (trainset을 순서대로 배분해야 정상적으로 동작한다.)  
    # Train - 10000 images
    # Val   -  1611 images
    # Test  -  1000 images  

### Modification Point  

* ```main.py```에서 실제 값과 예상값의 MSE(Mean Squared Error)를 구하는 부분이 있는데 따로 정의되지 않아 오류가 난다.  
따라서 sklearn.metrics의 mean_squared_error를 import해 주었다.  

* train_loss와 val_loss가 출력되고 따로 저장되지 않아 각각 csv 파일을 만들어준뒤 저장하는 구문을 추가했다.  

* cv2.imshow()를 cv2.imwrite()으로 바꿔주고 result_images에 저장하도록 변경했다.  