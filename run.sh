sudo ifconfig eth0 down
sudo ifconfig eth0 inet 192.168.88.24
sudo ifconfig eth0 up

sleep 3

python3 detector_ip.py
