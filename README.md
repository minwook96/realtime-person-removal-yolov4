# Disappearing-People - Person removal from complex backgrounds over time.
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
