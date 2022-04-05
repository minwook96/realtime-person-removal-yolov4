from onvif import ONVIFCamera
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler


class PTZ(object):
    def __init__(self):
        self.sche = BackgroundScheduler()
        self.media_profile = None
        self.status = False
        self.frame = None
        self.ptz = None
        self.x0 = -0.590399981
        self.y0 = -0.796000004
        self.x1 = 0.176599994
        self.y1 = -0.54369998
        self.x = self.x0
        self.y = self.y0
        self.distance = 0.1
        self.count = 0
        self.num = 15
        self.zoom = 0
        self.i = 1
        self.ptz_setting()

        # self.sche.start()

        # # 오전 9시 PTZ 실행
        # self.sche.add_job(self.ptz_control, 'cron', hour='09', minute='00', id='ptz1', args=[0.0])
        # self.sche.add_job(self.ptz_control, 'cron', hour='09', minute='05', id='ptz2', args=[self.num])
        #
        # # 오후 6시 PTZ 실행
        # self.sche.add_job(self.ptz_control, 'cron', hour='18', minute='00', id='ptz7', args=[0.0])
        # self.sche.add_job(self.ptz_control, 'cron', hour='18', minute='05', id='ptz8', args=[self.num])

    # PTZ 카메라 기본 변수 선언
    def ptz_setting(self):
        cam = ONVIFCamera("192.168.88.23", 80, "admin", "tmzkdl123$")
        media = cam.create_media_service()
        self.ptz = cam.create_ptz_service()
        self.media_profile = media.GetProfiles()[0]

        # 사용자 설정 프리셋 투어
        preset = self.ptz.GetPresets({
            'ProfileToken': self.media_profile.token
        })
        for i in range(0, len(preset)):
            print(preset[i].token)
            self.ptz.GotoPreset({
                'ProfileToken': self.media_profile.token,
                'PresetToken': preset[i].token
            })
            if self.i % 3 == 0:
                sleep(10)
                print(self.i)
            else:
                sleep(5)
                print(self.i)
            self.i = self.i + 1
        self.ptz.GotoHomePosition({'ProfileToken': self.media_profile.token})

    # def ptz_control(self, zoom):
    def ptz_control(self):
        while self.ptz is not None:
            self.ptz.AbsoluteMove({
                'ProfileToken': self.media_profile.token,
                'Position': {
                    'PanTilt': {
                        'x': self.x,
                        'y': self.y
                    },
                    'Zoom': {
                        'x': self.zoom
                    }
                }
            })

            sleep(10)

            # PTZ 제어 방향 [->]
            if self.count % 2 == 0:
                if self.x >= self.x1 and self.y <= self.y1:
                    self.y = self.y + self.distance
                    self.count = self.count + 1
                    self.i = self.i + 1

                elif self.x >= self.x1 and self.y >= self.y1:
                    self.x = self.x0
                    self.y = self.y0
                    self.count = 0
                    self.ptz.AbsoluteMove({
                        'ProfileToken': self.media_profile.token,
                        'Position': {
                            'PanTilt': {
                                'x': self.x,
                                'y': self.y
                            }
                        }
                    })

                    if self.zoom > 0.0:
                        self.ptz.GotoHomePosition({'ProfileToken': self.media_profile.token})
                        self.i = 0
                        self.zoom = 0
                        self.distance = 0.1
                    else:
                        self.zoom = 5
                        self.distance = 0.1 / self.zoom
                        self.zoom = 0.03125 * self.zoom
                else:
                    self.x = self.x + self.distance
                    self.i = self.i + 1
                    if self.x > self.x1:
                        self.x = self.x1

            # PTZ 제어 방향 [<-]
            elif self.count % 2 == 1:
                if self.x <= self.x0 and self.y <= self.y1:
                    self.y = self.y + self.distance
                    self.count = self.count + 1
                    self.i = self.i + 1

                elif self.x <= self.x0 and self.y >= self.y1:
                    self.x = self.x0
                    self.y = self.y0
                    self.count = 0
                    self.ptz.AbsoluteMove({
                        'ProfileToken': self.media_profile.token,
                        'Position': {
                            'PanTilt': {
                                'x': self.x,
                                'y': self.y
                            }
                        }
                    })
                    if self.zoom > 0.0:
                        self.ptz.GotoHomePosition({'ProfileToken': self.media_profile.token})
                        self.i = 0
                        self.zoom = 0
                        self.distance = 0.1
                    else:
                        self.zoom = 5
                        self.distance = 0.1 / self.zoom
                        self.zoom = 0.03125 * self.zoom

                else:
                    self.x = self.x - self.distance
                    self.i = self.i + 1
                    if self.x < self.x0:
                        self.x = self.x0


if __name__ == '__main__':
    ptz = PTZ()