import cv2
import numpy as np
import copy
import sys
sys.path.insert(0,'/home/mic-720ai/env1/humanless')
import darknet

import time
import itertools
from server import *

WIDTH = 1920
HEIGHT =1080


def intersection(rectA, rectB): # check if rect A & B intersect

	_a, _b = rectA, rectB
	a = [_a[0]-_a[2]/2 , _a[1] -_a[3]/2 , _a[0] + _a[2]/2 , _a[1] +_a[3]/2]
	b = [_b[0]-_b[2]/2 , _b[1] -_b[3]/2 , _b[0] + _b[2]/2 , _b[1] +_b[3]/2]
	startX = max( min(a[0], a[2]), min(b[0], b[2]) )
	startY = max( min(a[1], a[3]), min(b[1], b[3]) )
	endX = min( max(a[0], a[2]), max(b[0], b[2]) )
	endY = min( max(a[1], a[3]), max(b[1], b[3]) )
	if startX < endX and startY < endY:
		return True
	else:
		return False
def union(rectA, rectB): # create bounding box for rect A & B
	_a, _b = rectA, rectB
	a = [_a[0]-_a[2]/2 , _a[1] -_a[3]/2 , _a[0] + _a[2]/2 , _a[1] +_a[3]/2]
	b = [_b[0]-_b[2]/2 , _b[1] -_b[3]/2 , _b[0] + _b[2]/2 , _b[1] +_b[3]/2]
	startX = min( a[0], b[0] )
	startY = min( a[1], b[1] )
	endX = max( a[2], b[2] )
	endY = max( a[3], b[3] )
	#return [startX, startY, endX, endY]
	w = endX - startX
	h = endY - startY
	x = startX + w/2
	y = startY + h/2
	return [x,y,w,h]

def combine_boxes2(boxes):
	new_array = []
	for boxa, boxb in itertools.combinations(boxes, 2):
		if intersection(boxa[1], boxb[1]):
			new_array.append([boxa[0], union(boxa[1], boxb[1])])
		else:
			new_array.append(boxa)
	return new_array

def combine_boxes(rects):
	while (1):
		found = 0
		for ra, rb in itertools.combinations(rects, 2):
			if intersection(ra[1], rb[1]):
				if ra in rects:
					rects.remove(ra)
				if rb in rects:
					rects.remove(rb)
				rects.append([ra[0],union(ra[1], rb[1])])
				found = 1
				break
		if found == 0:
			break

	return rects 






####################################################################################
#Start Of unused functions -- used to fix the color difference problem but not work#
####################################################################################
def convert_color(source,target):
	def image_stats(image):
		# compute the mean and standard deviation of each channel
		l, a, b = cv2.split(image)
		lMean, lStd = (l.mean(), l.std())
		aMean, aStd = (a.mean(), a.std())
		bMean, bStd = (b.mean(), b.std())
		# return the color statistics
		return (lMean, lStd, aMean, aStd, bMean, bStd)
		
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")	
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar.astype(np.uint8)
	a -= aMeanTar.astype(np.uint8)		
	b -= bMeanTar.astype(np.uint8)		
	# scale by the standard deviations
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b
	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc
	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer

def hist_equal(img):
	hist, bins = np.histogram(img.flatten(), 256,[0,256])
	cdf = hist.cumsum()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	img2 = cdf[img]
	return img2


def convert_color_linear(target,ref_frame,results):
	def is_in_res(pointx,pointy,reses):
		for res in reses:
			if res[0] < x < res[0] + res[2] and	res[1] <y< res[1] + res[3]:
				return True
		return False
	def solve_equation(A,B):
		import numpy as np
		from itertools import combinations
		A = np.matrix(A)
		b = np.matrix(B)
		num_vars = A.shape[1]
		rank = np.linalg.matrix_rank(A)
		if rank == num_vars:				
			sol = np.linalg.lstsq(A, b)[0]	# not under-determined
		elif False:
			for nz in combinations(range(num_vars), rank):	# the variables not set to zero
				try: 
					sol = np.zeros((num_vars, 1))	
					sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
				except np.linalg.LinAlgError:	 
					pass	 
		return np.array(sol	)

	_h,_w,_c = ref_frame.shape
	res=	[]
	for i in results:
		w =int(i[1][2]*_w*1.2)
		h =int(i[1][3]*_h*1.2)
		x= int(i[1][0]*_w - w/2)
		y= int(i[1][1]*_h - h/2)
		res.append([x,y,w,h])
	import random
	random_points = []
	count = 0
	while len(random_points) < 200:
		#get random point
		x = random.randrange(0,_w)
		y = random.randrange(0,_h)
		count +=1
		if count > 5000:
			break
		if is_in_res(x,y,res) == False:
			random_points.append([x,y])
	showimg = target.copy()
	for x,y in random_points:
		showimg = cv2.circle(showimg,(x,y),2,(0,0,255),-1)
	cv2.imshow('qweqwe',showimg)
	
	lab_target = cv2.cvtColor(target,cv2.COLOR_BGR2LAB)
	lab_ref = cv2.cvtColor(ref_frame,cv2.COLOR_BGR2LAB)
	target_colors =[]
	ref_colors = []
	for x,y in	random_points:
		temp_lab = lab_target[y][x]
		target_colors.append( [temp_lab[0],temp_lab[1],temp_lab[2]])
		temp_lab = lab_ref[y][x]
		ref_colors.append([temp_lab[0],temp_lab[1],temp_lab[2]])
		## ref_color = target_color*M
	M = solve_equation(ref_colors,target_colors)
	result_image = np.rint(lab_target.dot(M)).astype('uint8')
	
	result = cv2.cvtColor(result_image,cv2.COLOR_LAB2BGR)
	
	cv2.imshow('Before',target)
	cv2.imshow('After',result)
	return result
def hist_match(source, template):

	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
																		return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)
##################################################################################
#End Of unused functions -- used to fix the color difference problem but not work#
##################################################################################

import Queue, threading, time

# bufferless VideoCapture
class IpVideoCapture:

	def __init__(self, name):
		self.cap = cv2.VideoCapture(name)
		self.q = Queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			#frame = frame[520:1079 ,480:1440 ] 
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()	 # discard previous (unprocessed) frame
				except Queue.Empty:
					pass
			self.q.put(frame)

	def read(self):
		return True, self.q.get()




class detector:
	def __init__(self,init_img,cam,factor=0.5):
	#	self.net = load_net("darknet/cfg/yolov4-tiny.cfg", "darknet/backup/yolov4-tiny.weights",0)
	#	self.meta = load_meta("darknet/cfg/voc.data")


		path = 'darknet/'
		cfg_path= path+'cfg/yolov4-tiny.cfg'
   	 	weight_path=path + 'backup/v4tiny_last_210311.weights' ## Hyndai
		#weight_path=path + 'backup/yolov4-tiny.weights' ##COCO data
		meta_path = path + 'cfg/coco.data'
		self.net = darknet.load_net_custom(cfg_path.encode('utf-8'),weight_path.encode('utf-8'),0,1 )
		self.darknet_image = darknet.make_image(darknet.network_width(self.net),
																  darknet.network_height(self.net),3)
		self.meta = darknet.load_meta(meta_path.encode('utf-8'))

		self.prev_frame = init_img

		self.blending_factor = factor

		self.prev_results=[]

		self.cam = cam
		self.factor = 2
		self.finish = False
		self.result = init_img	
                self.origin = init_img
		#self.mot_tracker=  Sort(max_age=1, min_hits=3, iou_threshold=0.3)

	def detect_img(self,img):
		frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,
																 (darknet.network_width(self.net),
																  darknet.network_height(self.net)),
																 interpolation=cv2.INTER_LINEAR)
		darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
		meta_results = darknet.detect_image(self.net,self.meta,self.darknet_image,thresh=0.1)


		results = []
		h,w,c = frame_resized.shape
		for i in meta_results:
			if i[0] == 'person':
				results.append([i[0],[i[2][0]/w , i[2][1]/h , i[2][2]/w , i[2][3]/h	]])

		return results
	def is_human(self,img):
		
		meta_results = detect_np(self.net,self.meta,img,thresh = 0.02) 
		for i in meta_results:
			if i[0] =='person':
				return True
		return False	

	def detect_imgs(self,imgs):
		results = []
		for img in imgs:
			results.append(self.detect_img(img))
		return results
	def track(self,frame,results):
		_h,_w,_c = frame.shape
		res = []
		for i in results:
			w =int(i[1][2]*_w)
			h =int(i[1][3]*_h)
			x= int(i[1][0]*_w - w/2)
			y= int(i[1][1]*_h - h/2)

			self.tracker.init(self.prev_frame, (x,y,w,h))
			(success , box) =tracker_res = self.tracker.update(frame)
			if success:
				(_x, _y, _w, _h) = [int(v) for v in box]
				res.append([_x+_w/2 , _y+_h/2, _w , _w,_h])
			else:
				res.append([x+w/2 , y+h, w/2 , _w,_h])
			
		return res
	def draw_result(self,frame,results):
		_h,_w,_c = frame.shape
		for i in results:
			w =int(i[1][2]*_w)
			h =int(i[1][3]*_h)
			x= int(i[1][0]*_w - w/2)
			y= int(i[1][1]*_h - h/2)
	
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
		return frame
	def update_prev_frame(self,frame,results):
		_h,_w,_c = frame.shape
		#results = combine_boxes(copy.deepcopy(results))
		prev_frame = self.prev_frame.copy()
		for i in results:
			w =int(i[1][2]*_w*self.factor)
			h =int(i[1][3]*_h*self.factor)
			x= int(i[1][0]*_w - w/2)
			y= int(i[1][1]*_h - h/2)
			if x < 0:
				x=0
			elif x+h > _w:
				w =_w - x
			if y < 0:
				y =0
			elif y+h > _h:
				h = _h - y	
			croped= prev_frame[y:y+h , x:x+w] 	
			frame[y:y+h , x:x+w] = croped
		temp = time.time()
	
		#arg1 = self.blending_factor*(np.array(frame).astype(np.uint8))
		#arg2 = (64 - self.blending_factor)*(np.array(self.prev_frame).astype(np.uint8))


		#self.prev_frame =((np.add(arg1,arg2))/64).astype(np.uint8)
	#	self.prev_frame =((np.add(arg1,arg2))/16).astype(np.uint8)
		self.prev_frame = cv2.addWeighted(  frame.copy() ,self.blending_factor , self.prev_frame.copy(),1-self.blending_factor ,0)
		#print(3, time.time()-temp)
#cv2.addWeighted(  frame.copy() ,self.blending_factor , self.prev_frame.copy(),1-self.blending_factor ,0)
	def savelog(self,frame , results):
		_h,_w,_c = frame.shape
		xs = []
		ys = []
		for i in results:
			w =int(i[1][2]*_w*self.factor)
			h =int(i[1][3]*_h*self.factor)
			x= int(i[1][0]*_w - w/2)
			y= int(i[1][1]*_h - h/2)
			if x < 0:
				x=0
			elif x+h > _w:
				w =_w - x
			if y < 0:
				y =0
			elif y+h > _h:
				h = _h - y
			xs.append(x)
			ys.append(y)
		if len(xs) ==0:
			return
		cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) 
		print("#Human Detected  {} #".format(cur_time))
		for idx, (x,y) in enumerate(zip(xs,ys)):
			print("Person {} location : {}, {}".format(idx+1,x,y))

	def main_loop(self):
		count =0
		loging_tick = time.time()
		while True:
			tick = time.time()
			ret, img = self.cam.read()
			if ret == False:
				continue
			img = cv2.resize(img,(WIDTH,HEIGHT))
			res = self.detect_img(img)
			cur_res =copy.deepcopy(res)
			#if time.time() - loging_tick > 1: 
			#	self.savelog(img,cur_res)
			#	loging_tick = time.time()
			for t in self.prev_results:
				for i in t:
					res.append(i)
			meta_result = img.copy()
			#meta_result = detector.draw_result(img.copy(),cur_res)# a result that detects person
			self.origin = meta_result
			self.update_prev_frame(img.copy(),res)
			wrresult = cv2.hconcat([meta_result,self.prev_frame])
			self.result= wrresult
			self.result= self.prev_frame
		#	cv2.imshow('Result',wrresult)
		#		out1.write(wrresult)
				#out2.write(result)
		#		cv2.imshow('Result Inpainting',result1)
			if len(res) != 0:
				detector.prev_results.append(cur_res)
			while len(detector.prev_results) >= 60:
				detector.prev_results.pop(0)
			time_col = time.time() - tick
		#	print('{}	Sec	 =>	 {} FPS'.format(time_col , round(1./time_col),3))

		


	
if __name__ == '__main__':
#	cam = cv2.VideoCapture(0)
	#cam = IpVideoCapture('rtsp://admin:119@mogsol@169.254.167.60/profile2/media.smp')
	#cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	#cam = cv2.VideoCapture('media/3.mp4')
	#cam = IpVideoCapture('rtsp://admin:119@mogsol@169.254.167.60/profile2/media.smp')
	#camurl="rtsp://admin:admin4321@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif" 
	#camurl="rtsp://admin:4321@192.168.1.101:554/profile2/media.smp" 
        camurl="rtsp://admin:1q2w3e4r5t@192.168.1.101:554/profile2/media.smp"

        
	blending_factor = 0.5
	######USAGE : rtsp://USERID:PASSWORD@URL)
	cam = IpVideoCapture(camurl)
	#ret, img = cam.read()
	#cv2.imshow('qweqwewqe',img)
	#cv2.waitKey(0)
	
	#val, frame = cam.read()
	init_image=np.zeros((HEIGHT,WIDTH,3),np.uint8)
	init_image[:,:]=(0,0,0)
#	init_image = cv2.imread('background.png')
#	init_frame = cv2.resize(init_image,(600,400))
	detector = detector(init_image,cam,blending_factor)	
	t = threading.Thread(target=detector.main_loop)
	t.daemon = True
	t.start()

	#########LAUNCH RTSP SERVER################
	origin_server = GstServer(sub_dir ='/origin',port = 8554)
	origin_server.run()

	filtered_server = GstServer(sub_dir = '/filter',port = 8555)
	filtered_server.run()
	while True:
		img = detector.origin
		origin_server.set_img(img.copy())
		img = detector.result
		filtered_server.set_img(img.copy())
		time.sleep(0.05)
	
	count =0
	#cv2.namedWindow(10.150.160.178'Result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
