# Tensorflow_Tacotron1_TTS
Tacotron1과 음성데이터를 이용한 TTS만들기


참조  
chldkato님 gitgub  
https://github.com/chldkato/Tacotron-Korean-Tensorflow2
  
  
## 1. 실행환경 만들기
### 1-1 Anaconda3 설치 및 가상환경 만들기
Anaconda3(https://www.anaconda.com/products/individual) 사이트에 들어가서 중간쯤 내리면 다운로드 페이지가 나오는데 본인의 환경에 맞는 Anaconda3를 다운로드 및 설치를 해줍니다.

설치가 완료되면 Anaconda Prompt를 실행한 후 가상환경을 만들어 줍니다.  
```
conda create -n tensorflow python=3.8
```
여기서는 tensorflow라는 이름의 가상환경을 파이썬 버전 3.8로 하여 만들겠습니다.
  
  
가상환경이 만들어지면 
```
activate tensorflow
```
명령을 사용해 가상환경으로 들어가줍니다.  
prompt에 (base)가 아닌 (tensorflow)로 시작하게 바뀌었다면 제대로 들어온 것입니다.
  
  
## 2. 학습에 필요한 데이터셋 만들기
이 코드에서는 Kaggle에서 제공하는 KSS데이터 셋을 다운받아 사용했습니다.  (https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)  
다운로드 받아 압축을 푼 후 코드가 있는 프로젝트 폴더에 이미지와 같이 넣어주시면 됩니다.   
 ```
   Tensorflow_Tacotron1_TTS
     |- kss
         |- 1
         |- 2
         |- 3
         |- 4
         |- transcript.v.1.4.txt
   ```
  
## 3. 전처리
anaconda prompt에서 가까 만들었던 가상환경으로 접속한 후 코드가 있는 폴더로 이동해 줍니다.  
  ```
  cd 코드가 있는 파일 경로
  ```
  
이동했으면 train1.py코드를 싱행시켜서 학습해 줍니다.  
```
python train1.py
```
  
만약 실행시 학습에 필요한 모듈이 없다고 나올때는 anaconda prompt상에서 가상환경에 들어간 후 아래 명령을 실행하면 설치가 진행됩니다.
```
pip install 필요한 모듈명
ex) pip install pandas
```  

전처리가 모두 진행되면  
 ```
   Tensorflow_Tacotron1_TTS
     |- data
         |- dec
         |- mel
         |- spec
         |- text
         |- mel_len.npy
         |- text_len.npy
   ```
위와 같이 data폴더에서 학습데이터가 잘 생성되었음을 알 수 있습니다.
  
  
## 4. 학습 진행
### 4-1 임배딩 ~ 디코더 학습

아래와 같은 코드를 사용하여 학습을 시작합니다.  
util/hparams.py 안에 있는 checkpoint_step만큼 학습이 진행 될때마다 학습모델과 mel-spectrogram 이미지가 checkpoint/1에 저장되며    
멈추고 싶을떄는 Ctrl + c 를 사용하여 중지할 수 있습니다.  
```
python train1.py
```
  
마찬가지로 필요한 라이브러리들을 import하고 없으면 pip를 통해 설치하면 되는데 여기서 tensorflow환경 설정이 중요합니다.
만약 본인이 그래픽 카드가 없다고 한다면 cpu로만 진행해야 하기 때문에 anaconda prompt에서 아래 명령을 통해 cpu버전의 tensorflow를 설치해줍니다.
```
pip install --ignore-installed --upgrade tensorflow-cpu
```  
  
그래픽카드를 사용하게 되면 아래 명령을 통해 설치하시고 
```
pip install tensorflow
```  
CUDA와 CUDNN을 별도로 설치 해주셔야 진행이 되는데 두개의 버전을 잘못 맞추면 실행이 안되니 인터넷에 설명을 잘 읽어보고 설치하셔야 합니다.  
ex) https://blog.naver.com/shino1025/222408513746
  
  
checkpoint/1에서 생성되는이미지를 보면서 어느정도 학습이 된 것 같으면 중지하고 다음 학습으로 넘어가도 되겠습니다.  
10만 epoch 이상을 권장합니다.  
#### 학습 예시
* 500 epoch  
<img src = "./ScreenShots/1.png" width="60%">  
* 3500 epoch  
<img src = "./ScreenShots/2.png" width="60%">  
* 9000 epoch  
<img src = "./ScreenShots/3.png" width="60%">  
* 169000 epoch  
<img src = "./ScreenShots/4.png" width="60%">
  
### 4-2 Attention 학습
아래 코드를 이용하여 4-2과 마찬가지로 학습을 진행하면 되고  
역시 똑같이 checkpoint_step만큼 학습이 진행될때마다 모델이 checkpoint/2에 저장됩니다.  
```
python train2.py
```
마찬가지로 학습이 어느정도 진행되었다면 정지버튼을 눌러 종료해주시면 됩니다.  
5만 epoch 이상을 권장합니다.  

## 5. 테스트
test.py 코드 안 sentences를 만들고 싶은 말로 바꾸어 저장해줍니다.  (특수문자 제외 한글만)  
```
sentences = ['안녕하세요 반갑습니다']
```
test코드를 실행해 줍니다.  
```
python test.py
```
실행이 완료되면 output폴더에 alignment이미지와 npy파일, wav파일이 생성됩니다.
 ```
   Tensorflow_Tacotron1_TTS
     |- output
         |- 0.wav
         |- align-0.png
         |- mel-0.npy
   ```
   
이 예제로 생성된 음성파일은 바로 재생이 불가능하여 업로드 해두었습니다.  
