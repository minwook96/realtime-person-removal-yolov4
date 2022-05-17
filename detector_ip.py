from server import *
import numpy as np
import itertools
import threading
import darknet
import random
import queue
import time
import copy
import sys
import cv2

sys.path.insert(0, '/home/skysys/humanless')

WIDTH = 1920
HEIGHT = 1080


def intersection(rect_a, rect_b):  # check if rect A & B intersect
    _a, _b = rect_a, rect_b
    a = [_a[0] - _a[2] / 2, _a[1] - _a[3] / 2, _a[0] + _a[2] / 2, _a[1] + _a[3] / 2]
    b = [_b[0] - _b[2] / 2, _b[1] - _b[3] / 2, _b[0] + _b[2] / 2, _b[1] + _b[3] / 2]
    start_x = max(min(a[0], a[2]), min(b[0], b[2]))
    start_y = max(min(a[1], a[3]), min(b[1], b[3]))
    end_x = min(max(a[0], a[2]), max(b[0], b[2]))
    end_y = min(max(a[1], a[3]), max(b[1], b[3]))
    if start_x < end_x and start_y < end_y:
        return True
    else:
        return False


def union(rect_a, rect_b):  # create bounding box for rect A & B
    _a, _b = rect_a, rect_b
    a = [_a[0] - _a[2] / 2, _a[1] - _a[3] / 2, _a[0] + _a[2] / 2, _a[1] + _a[3] / 2]
    b = [_b[0] - _b[2] / 2, _b[1] - _b[3] / 2, _b[0] + _b[2] / 2, _b[1] + _b[3] / 2]
    start_x = min(a[0], b[0])
    start_y = min(a[1], b[1])
    end_x = max(a[2], b[2])
    end_y = max(a[3], b[3])
    w = end_x - start_x
    h = end_y - start_y
    x = start_x + w / 2
    y = start_y + h / 2
    return [x, y, w, h]


def combine_boxes2(boxes):
    new_array = []
    for box_a, box_b in itertools.combinations(boxes, 2):
        if intersection(box_a[1], box_b[1]):
            new_array.append([box_a[0], union(box_a[1], box_b[1])])
        else:
            new_array.append(box_a)
    return new_array


def combine_boxes(rect):
    while True:
        found = 0
        for ra, rb in itertools.combinations(rect, 2):
            if intersection(ra[1], rb[1]):
                if ra in rect:
                    rect.remove(ra)
                if rb in rect:
                    rect.remove(rb)
                rect.append([ra[0], union(ra[1], rb[1])])
                found = 1
                break
        if found == 0:
            break
    return rect


# buffer less VideoCapture
class IpVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.frame_counter = 0
        read_thread = threading.Thread(target=self._reader)
        read_thread.daemon = True
        read_thread.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()

            # video
            self.frame_counter += 1
            # If the last frame is reached, reset the capture and the frame_counter
            if self.frame_counter == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.frame_counter = 0  # Or whatever as long as it is the same as next line
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # live
            # if not ret:
            # 	break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return True, self.q.get()


class Detector:
    def __init__(self, init_img, camera, factor=0.5):
        path = '/home/skysys/humanless/darknet/'
        cfg_path = path + 'cfg/yolov4-csp.cfg'
        weight_path = path + 'backup/yolov4-csp.weights'
        # cfg_path = path + 'cfg/yolov4-custom-csp.cfg'
        # weight_path = path + 'backup/yolov4-custom-csp.weights'
        meta_path = path + 'cfg/coco.data'
        self.network, self.class_names, self.colors = darknet.load_network(cfg_path, meta_path, weight_path)
        self.darknet_image = darknet.make_image(darknet.network_width(self.network),
                                                darknet.network_height(self.network), 3)
        self.prev_frame = init_img
        self.blending_factor = factor
        self.prev_results = []
        self.cam = camera
        self.factor = 2
        self.finish = False
        self.result = init_img
        self.origin = init_img

    def detect_img(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.network), darknet.network_height(self.network)),
                                   cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        meta_results = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.3)
        results = []
        h, w, c = frame_resized.shape

        for i in meta_results:
            if i[0] == 'person':
                results.append([i[0], [i[2][0] / w, i[2][1] / h, i[2][2] / w, i[2][3] / h]])

        return results

    def detect_imgs(self, images):
        results = []
        for image in images:
            results.append(self.detect_img(image))
        return results

    def track(self, frame, results):
        _h, _w, _c = frame.shape
        res = []
        for i in results:
            w = int(i[1][2] * _w)
            h = int(i[1][3] * _h)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)

            self.tracker.init(self.prev_frame, (x, y, w, h))
            (success, box) = self.tracker.update(frame)
            if success:
                (_x, _y, _w, _h) = [int(v) for v in box]
                res.append([_x + _w / 2, _y + _h / 2, _w, _w, _h])
            else:
                res.append([x + w / 2, y + h, w / 2, _w, _h])

        return res

    def draw_result(self, frame, results):
        _h, _w, _c = frame.shape
        for i in results:
            w = int(i[1][2] * _w)
            h = int(i[1][3] * _h)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)

            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return frame

    def mosaic(self, src, ratio=0.1):
        small = cv2.resize(src, None, fx=ratio, fy=ratio)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_AREA)

    # small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    # return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def mosaic_area(self, src, x, y, width, height, ratio=0.1):
        dst = src.copy()
        dst[y:y + height, x:x + width] = self.mosaic(dst[y:y + height, x:x + width], ratio)
        return dst

    # mosaic
    # def update_prev_frame(self, frame, results):
    # 	_h, _w, _c = frame.shape
    # 	for i in results:
    # 		w = int(i[1][2] * _w * self.factor)
    # 		h = int(i[1][3] * _h * self.factor)
    # 		x = int(i[1][0] * _w - w/2)
    # 		y = int(i[1][1] * _h - h/2)
    # 		if x < 0:
    # 			x = 0
    # 		elif x + h > _w:
    # 			w = _w - x
    # 		if y < 0:
    # 			y = 0
    # 		elif y + h > _h:
    # 			h = _h - y
    # 		self.result = self.mosaic_area(frame, x, y, w, h)

    # humanless
    def update_prev_frame(self, frame, results):
        _h, _w, _c = frame.shape
        results = combine_boxes(copy.deepcopy(results))
        prev_frame = self.prev_frame.copy()
        for i in results:
            w = int(i[1][2] * _w * self.factor)
            h = int(i[1][3] * _h * self.factor)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)
            # print("w, h, x, y", w, h, x, y)
            if x < 0:
                x = 0
            elif x + h > _w:
                w = _w - x
            if y < 0:
                y = 0
            elif y + h > _h:
                h = _h - y
            cropped = prev_frame[y:y + h, x:x + w]
            frame[y:y + h, x:x + w] = cropped

        self.prev_frame = cv2.addWeighted(frame.copy(), self.blending_factor, self.prev_frame.copy(),
                                          1 - self.blending_factor, 0)

    def save_log(self, frame, results):
        _h, _w, _c = frame.shape
        xs = []
        ys = []
        for i in results:
            w = int(i[1][2] * _w * self.factor)
            h = int(i[1][3] * _h * self.factor)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)
            if x < 0:
                x = 0
            elif x + h > _w:
                w = _w - x
            if y < 0:
                y = 0
            elif y + h > _h:
                h = _h - y

            xs.append(x)
            ys.append(y)

        if len(xs) == 0:
            return
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        print("#Human Detected  {} #".format(cur_time))
        for idx, (x, y) in enumerate(zip(xs, ys)):
            print("Person {} location : {}, {}".format(idx + 1, x, y))

    # mosaic
    # def main_loop(self):
    # 	while True:
    # 		ret, frame = self.cam.read()
    # 		if not ret:
    # 			continue
    # 		image = cv2.resize(frame, (WIDTH, HEIGHT))
    # 		res = self.detect_img(image)
    # 		for t in self.prev_results:
    # 			print(t)
    # 			for i in t:
    # 				res.append(i)
    # 		meta_result = image.copy()
    # 		self.origin = meta_result
    # 		self.result = meta_result
    # 		self.update_prev_frame(image.copy(), res)

    # humanless
    def main_loop(self):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_1000)
        parameters = cv2.aruco.DetectorParameters_create()
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue
            image = cv2.resize(frame, (WIDTH, HEIGHT))
            res = self.detect_img(image)
            cur_res = copy.deepcopy(res)
            # self.save_log(image, cur_res)
            for t in self.prev_results:
                for i in t:
                    res.append(i)
            meta_result = image.copy()
            # AR 마커 인식
            # gray = cv2.cvtColor(meta_result, cv2.COLOR_BGR2GRAY)
            # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            # meta_result = cv2.aruco.drawDetectedMarkers(meta_result.copy(), corners, ids)
            self.origin = meta_result
            self.update_prev_frame(image.copy(), res)
            wr_result = cv2.hconcat([meta_result, self.prev_frame])
            self.result = wr_result
            self.result = self.prev_frame
            if len(res) != 0:
                detector.prev_results.append(cur_res)
            while len(detector.prev_results) >= 60:
                detector.prev_results.pop(0)


if __name__ == '__main__':
    cam_url = "rtsp://admin:tmzkdl123$@192.168.88.23/profile2/media.smp"
    # cam_url = "./test3.mp4"

    blending_factor = 0.5
    cam = IpVideoCapture(cam_url)
    init_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    init_image[:, :] = (0, 0, 0)

    detector = Detector(init_image, cam, blending_factor)
    t = threading.Thread(target=detector.main_loop)
    t.daemon = True
    t.start()

    origin_server = GstServer(sub_dir='/origin', port=8554)
    origin_server.run()

    filtered_server = GstServer(sub_dir='/filter', port=8555)
    filtered_server.run()

    while True:
        img = detector.origin
        origin_server.set_img(img.copy())
        img = detector.result
        filtered_server.set_img(img.copy())
        time.sleep(0.05)
