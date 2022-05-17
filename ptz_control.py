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

        self.x_first = 0.8
        self.y_first = -0.6
        self.x_last = -0.7
        self.y_last = -0.2

        # 역방향
        # self.x_first = 1
        # self.y_first = 0.7
        # self.x_last = -0.7
        # self.y_last = 0.5

        # 정방향
        # self.x_first = -0.2
        # self.y_first = 0.7
        # self.x_last = 0.2
        # self.y_last = 0.5

        self.x = self.x_first
        self.y = self.y_first
        self.x_home = 0
        self.y_home = 0
        self.distance = 0.1
        self.count = 0
        self.num = 15
        self.zoom = 0
        self.i = 1
        self.ptz_setting()

        self.sche.start()

        # 오전 7시 PTZ 실행
        self.sche.add_job(self.tour_start, 'cron', hour='07', minute='00', id='ptz1')
        self.sche.add_job(self.ptz_control, 'cron', hour='07', minute='05', id='ptz2', args=[0.0])
        self.sche.add_job(self.ptz_control, 'cron', hour='07', minute='15', id='ptz3', args=[self.num])

        # 오후 6시 PTZ 실행
        self.sche.add_job(self.tour_start, 'cron', hour='18', minute='00', id='ptz4')
        self.sche.add_job(self.ptz_control, 'cron', hour='18', minute='05', id='ptz5', args=[0.0])
        self.sche.add_job(self.ptz_control, 'cron', hour='18', minute='15', id='ptz6', args=[self.num])

    # PTZ 카메라 기본 변수 선언
    def ptz_setting(self):
        cam = ONVIFCamera("192.168.88.23", 80, "admin", "tmzkdl123$")
        media = cam.create_media_service()
        self.ptz = cam.create_ptz_service()
        self.media_profile = media.GetProfiles()[0]
        # print(self.ptz.GetStatus({
        #     'ProfileToken': self.media_profile.token,
        # }))

    def tour_start(self):
        preset = self.ptz.GetPresets({
            'ProfileToken': self.media_profile.token
        })

        for i in range(0, len(preset)):
            self.ptz.GotoPreset({
                'ProfileToken': self.media_profile.token,
                'PresetToken': preset[i].token
            })
            if self.i % 3 == 0:
                sleep(15)
            else:
                sleep(10)
            self.i = self.i + 1
        self.ptz.GotoHomePosition({'ProfileToken': self.media_profile.token})

    def ptz_control(self, zoom):
        self.zoom = zoom
        if self.zoom > 0:
            self.distance = 0.1 / self.zoom
            self.zoom = 0.03125 * self.zoom
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

            sleep(2)

            # 정방향
            if self.x_first < self.x_last:
                # PTZ 제어 방향 [->]
                if self.count % 2 == 0:
                    if self.x >= self.x_last and self.y <= self.y_last:
                        print("y축 이동")
                        self.y = self.y + self.distance
                        self.count = self.count + 1
                        print("y :", self.y)

                    elif self.x >= self.x_last and self.y >= self.y_last:
                        self.x = self.x_first
                        self.y = self.y_first
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
                            self.zoom = 0
                            self.distance = 0.1
                            break
                    else:
                        print("정방향 x축 이동 [->]")
                        self.x = self.x + self.distance
                        if self.x > self.x_last:
                            self.x = self.x_last
                        print("x :", self.x)

                # PTZ 제어 방향 [<-]
                else:
                    if self.x <= self.x_first and self.y <= self.y_last:
                        print("y축 이동")
                        self.y = self.y + self.distance
                        self.count = self.count + 1
                        print("y :", self.y)

                    elif self.x <= self.x_first and self.y >= self.y_last:
                        self.x = self.x_first
                        self.y = self.y_first
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
                            self.zoom = 0
                            self.distance = 0.1
                            break
                    else:
                        print("정방향 x축 이동 [<-]")
                        self.x = self.x - self.distance
                        if self.x < self.x_first:
                            self.x = self.x_first
                        print("x :", self.x)
            # 역방향
            else:
                # PTZ 제어 방향 [->]
                if self.count % 2 == 0:
                    if self.x_last <= self.x and not(self.x_first <= self.x <= 1.0) and self.y_last >= self.y:
                        print("y축 이동")
                        self.y = self.y + self.distance
                        self.count = self.count + 1
                        print("y :", self.y)

                    elif (self.x_first <= self.x <= 1.0 or -1.0 <= self.x <= self.x_last) and self.y_last >= self.y:
                        print("역방향 x축 이동 [->]")
                        self.x = self.x + self.distance
                        if self.x >= 1.0 or self.x <= -1.0:
                            self.x = -1.0
                        elif self.x_last < self.x and not(self.x_first <= self.x <= 1.0):
                            self.x = self.x_last
                        print("x :", self.x)

                    elif self.x >= self.x_last and self.y >= self.y_last:
                        print("return")
                        self.x = self.x_first
                        self.y = self.y_first
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
                            self.zoom = 0
                            self.distance = 0.1
                            break
                        else:
                            self.ptz.GotoHomePosition({'ProfileToken': self.media_profile.token})
                            self.zoom = 0
                            self.distance = 0.1
                            break

                # PTZ 제어 방향 [<-]
                else:
                    if self.x <= self.x_first and self.y >= self.y_last:
                        print("return")
                        self.x = self.x_first
                        self.y = self.y_first
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
                            self.zoom = 0
                            self.distance = 0.1
                            break
                        else:
                            self.ptz.GotoHomePosition({'ProfileToken': self.media_profile.token})
                            self.zoom = 0
                            self.distance = 0.1
                            break

                    elif self.x <= self.x_first and not(-1.0 <= self.x <= self.x_last) and self.y_last > self.y:
                        print("y축 이동")
                        self.y = self.y + self.distance
                        self.count = self.count + 1
                        print("y :", self.y)

                    elif (-1.0 <= self.x <= self.x_last or self.x_first <= self.x <= 1.0) and self.y_last > self.y:
                        print("역방향 x축 이동 [<-]")
                        self.x = self.x - self.distance
                        if self.x >= 1.0 or self.x <= -1.0:
                            self.x = 1.0
                        elif self.x < self.x_first and not(-1.0 <= self.x <= self.x_last):
                            self.x = self.x_first
                        print("x :", self.x)


if __name__ == '__main__':
    ptz = PTZ()
    ptz.ptz_control(15)