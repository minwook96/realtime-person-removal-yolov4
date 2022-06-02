# Person removal from complex backgrounds over time.
Removing people from complex backgrounds in real time using Darknet(YoloV4) using Python.

## Install
1. Install Darknet (https://github.com/AlexeyAB/darknet) \
cd darknet \
make -j 8 \
open darknet.py and change libdarknet.so's path

2. Install requirements.txt \
pip3 install -r requirements.txt

3. Install gi \
sudo apt install libcairo2-dev \
sudo apt install libxt-dev \
sudo apt install libgirepository1.0-dev \
pip3 install pycairo \
pip3 install PyGObject

## RUN program
python3 detector.py


## CCTV PTZ Control(Onvif)
your cctv url insert
select PTZ Control

![그림1](https://user-images.githubusercontent.com/49277505/137436287-adf48b59-389d-4ff2-8d5c-a1bcc977e766.png)


    self.x0 = 0.534099996 
    self.y0 = -0.851100028  # A
    self.x1 = 0.734099996
    self.y1 = -0.551100028  # B
    mycam = ONVIFCamera("192.168.88.14", 80, "admin", "tmzkdl123$") # your cctv url
    

## original (static cctv) (1920x1080 FHD)

![210330_원본영상 mp4_20211015_135512](https://user-images.githubusercontent.com/49277505/137435545-01bfc1d7-2eef-4f0f-9d02-9d1bcf84a18f.gif)

## real-time CCTV Streamin remove people (1920x1080 FHD)

![210330_삭제영상 mp4_20211015_135254](https://user-images.githubusercontent.com/49277505/137435541-7a52fd28-0795-4b32-8d6b-67b697470f40.gif)

## Usage

Feel free to use in your own projects. Code is released under python licence. If you decide to use my code please consider giving me a shout out! Would love to see what others create with it :-) Thanks.
