import tkinter  # Tkinter 및 GUI 관련
import tkinter.ttk
import PIL.Image, PIL.ImageTk

import threading  # Thread

result_img = 0  # 전역변수로 최종 이미지를 받도록 했다


class App(threading.Thread):
    def __init__(self, window, window_title):
        threading.Thread.__init__(self)
        self.window = window
        self.window.title(window_title)
        self.delay = 15

        # View Video
        self.canvas = tkinter.Canvas(window, width=608, height=480)
        self.canvas.pack()

        self.update()
        self.window.mainloop()

    def update(self):
        try:
            vid = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(vid))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        except:
            pass
        self.window.after(self.delay, self.update)


def main():
    global result_img
    vid = cv2.VideoCapture(0)  # 비디오 받음
    video_frame_cnt = int(vid.get(7))

    with tf.Session() as sess:  # Session은 여기서 열자
        # Session Open

        for i in range(video_frame_cnt):
            ret, img_ori = vid.read()

            # 영상에서 객체 검출이나 다른 알고리즘 등등...

            result_img = cv2.resize(img_ori, (640, 480))  # 결과 이미지를 넘겨주자

        vid.release()


if __name__ == "__main__":
    t1 = threading.Thread(target=main, args=())
    t1.daemon = True
    t1.start()
    t2 = App(tkinter.Tk(), "GUI")
    t2.daemon = True
    t2.start()