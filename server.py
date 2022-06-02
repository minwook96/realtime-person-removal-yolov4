import gi
import threading
import socket
import fcntl
import struct
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


def get_ip_address(ifname):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15], 'utf-8')))[20:24])


class SensorFactory(GstRtspServer.RTSPMediaFactory):
	def __init__(self, **properties):
		super(SensorFactory, self).__init__(**properties)
		self.number_frames = 0
		self.fps = 15
		self.duration = 1. / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
		self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,' \
							 'format=BGR,width=1920,height=1080,framerate={}/1 ! videoconvert ! video/x-raw,' \
							 'format=I420 ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay ' \
							 'config-interval=1 name=pay0 pt=96'.format(self.fps)
		self.img = None

	def on_need_data(self, src, lenght):
		if True:
			ret = True
			frame = self.img
			if ret:
				data = frame.tostring()
				buf = Gst.Buffer.new_allocate(None, len(data), None)
				buf.fill(0, data)
				buf.duration = self.duration
				timestamp = self.number_frames * self.duration
				buf.pts = buf.dts = int(timestamp)
				buf.offset = timestamp
				self.number_frames += 1
				retval = src.emit('push-buffer', buf)
				if retval != Gst.FlowReturn.OK:
					print(retval)

	def do_create_element(self, url):
		return Gst.parse_launch(self.launch_string)

	def do_configure(self, rtsp_media):
		self.number_frames = 0
		appsrc = rtsp_media.get_element().get_child_by_name('source')
		appsrc.connect('need-data', self.on_need_data)
		print('#Connected')

	def set_img(self, img):
		self.img = img


class GstServer(GstRtspServer.RTSPServer):
	def __init__(self, sub_dir, port, **properties):
		GObject.threads_init()
		Gst.init(None)
		self.factory = SensorFactory()
		self.factory.set_shared(True)
		self.server = GstRtspServer.RTSPServer()
		self.server.get_mount_points().add_factory(sub_dir, self.factory)
		self.server.set_service(str(port))

		auth = GstRtspServer.RTSPAuth()
		token = GstRtspServer.RTSPToken()
		token.set_string('media.factory.role', "admin")
		basic = GstRtspServer.RTSPAuth.make_basic("admin", "tmzkdl123")
		auth.add_basic(basic, token)
		self.server.set_auth(auth)

		permissions = GstRtspServer.RTSPPermissions()
		permissions.add_permission_for_role("admin", "media.factory.access", True)
		permissions.add_permission_for_role("admin", "media.factory.construct", True)

		self.factory.set_permissions(permissions)
		self.server.attach(None)
		cur_ip = get_ip_address('eth0')
		print("current ip : ", cur_ip)
		print('#Gst Server: rtsp://admin:tmzkdl123@{}:{}{}'.format(cur_ip, port, sub_dir))

	def run(self):
		loop = GObject.MainLoop()
		t = threading.Thread(target=loop.run)
		t.daemon = True
		t.start()

	def set_img(self, img):
		self.factory.set_img(img)


if __name__ == '__main__':
	GObject.threads_init()
	Gst.init(None)

	server = GstServer()

	loop = GObject.MainLoop()
	loop.run()