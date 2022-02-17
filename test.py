import cv2
import queue
import threading


class IpVideoCapture:
	def __init__(self, name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()

	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()	 # discard previous (unprocessed) frame
				except queue.Empty:
					pass
			self.q.put(frame)

	def read(self):
		return True, self.q.get()


camurl = "rtsp://admin:tmzkdltltm123@192.168.88.25:554/profile2/media.smp"

cam = cv2.VideoCapture(camurl)
print('Ip Cam Port opened')
while True:
	ret, frame = cam.read()
	if ret:
		cv2.imshow('qwe', frame)
	key = cv2.waitKey(1)
	if str(key) == 'q':
		break
