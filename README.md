# 실전 추천시스템 알고리즘 이해와 구현

## 맺음말

3일간 밀도 높은 강의를 열심히 들어주셔서 감사합니다. 고생 많으셨습니다. 

강의 내용이나 Data Science 관련해서 고민있으시면 연락주세요~

내일 크리스마스이네요.  **메리 크리스마스** 입니다. 그리고 새해 복 많이 받으세요.

## 강사

멀티캠퍼스 강선구(sunku0316.kang@multicampus.com, sun9sun9@gmail.com)

## 기본 환경 구성

Windows 11 / Docker powered by WSL Ubuntu(24.04) / NVIDIA GPU


### NVIDIA 드라이브 업데이트

- NVIDIA 장비 확인
시스템 > 정보 > 고급 시스템 설정

- NVIDIA 드라이버 업데이트

강의장 PC 기준

https://www.nvidia.com/ko-kr/drivers/details/241094/

설치 파일 다운로드 

실행 > NVIDIA 그래픽 드라이버 > 빠른 설치

### WSL 업데이트

- WSL을 업데이트 합니다.

```cmd
wsl --update
```
```cmd
wsl --install --distribution Ubuntu-24.04
```
아이디 패스워드 프롬프트 창에서

```
아이디/패스워드 = multi_rcmd / multi_rcmd 
```

기본적으로 Ubunto-24.04 환경이 되도록 합니다.
```cmd
wsl --set-default Ubuntu-24.04
```


설정

WSL 구동

```cmd
wsl
```

### 로컬 경로 생성

wsl 환경에 c 드라이브 밑에 multi_rcmd 폴더를 생성합니다.

```bash
mkdir /mnt/c/multi_rcmd 
```

### docker

install_docker.sh 파일과 docker-compose.yml을 다운로드 받고, 파일을 C:\multi_rcmd 경로에 위치 시킵니다.


- multi_rcmd 폴더로 이동

```bash
cd /mnt/c/multi_rcmd
```

```bash
sudo chmod 755 install_docker.sh
sudo ./install_docker.sh
sudo usermod -aG docker $USER
```

wsl을 종료 시킨후
```bash
exit
```

다시 wsl에 접근하면, docker를 sudo 없이 사용할 수 있게 됩니다.
```bash
wsl
```

### 수업자료 다운로드

**강의장에서는 공유드라이브를 통해 자료를 공유할 예정입니다. 개별적으로 구성하실 경우 아래 경로에 있는 폴더들을 다운로드 받아주세요**

[Google Drive](https://drive.google.com/drive/folders/1B2MWhhEjf1HChP85n9mp8Bp-UvqdvLLA)

#### 폴더별 자료 구성

- docker: docker 이미지
- model: 추천 모델
- dataset: 데이터셋

### Docker 이미지 탑재

커멘드 창을 열고 docker 이미지가 있는 경로로 이동합니다.

1. Oracle Image 탑재

```
docker load -i oracle-rcmd.tar
```

2. Qdrant Image 탑재

```
docker load -i qdrant_rcmd.tar
```

3. 실습 환경 탑재

```
docker load -i multi_rcmd.tar
```


## 실습 환경 구동

실행창에서 wsl 환경을 구동시킵니다.

```
wsl
```

wsl에서 실습 폴더로 이동합니다. 

```
cd /mnt/c/multi_rcmd
```

docker-compose.yml 파일을 다운로드 받고, 실습 메인 경로(C:\multi_rcmd)  위치시킵니다.

실습 환경 시작
```cmd
docker compose up -d
```

실습 환경 종료
```cmd
docker compose down
```

### Jupyter lab에 접근

URL: http://localhost:8888


### 영화 추천 데모 웹앱 접근

URL: http://localhost:5001

