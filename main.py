import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# Globals
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Rect:
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

# creates a thot detector in c to speed up their wrangling
def fork_thot():
    pass

def get_thots(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_rects = face_detector.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5)

    rects = []
    for face_rect in face_rects:
        x0 = face_rect[0]
        y0 = face_rect[1]
        x1 = x0 + face_rect[2]
        y1 = y0 + face_rect[3]
        rects.append(Rect(Point(x0, y0), Point(x1, y1)))
    return rects

def draw_rect(image, color, rect):#                                             thickness
    cv2.rectangle(image, (rect.p0.x, rect.p0.y), (rect.p1.x, rect.p1.y), color, 5)

def draw_text(image, color, pt, text): #                             scale       thickness
    cv2.putText(image, text, (pt.x, pt.y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 5, cv2.LINE_AA)

def draw_thot(image, thot_rect):
    draw_rect(image, (255, 255, 0), thot_rect)
    draw_text(image, (0, 255, 255), Point(thot_rect.p1.x - 100, thot_rect.p1.y - 5), 'THOT')

# Returns an image with the thot overlay
def thot_overlay(image):
    overlayed_image = image.copy()
    thot_rects = get_thots(overlayed_image)

    for thot_rect in thot_rects:
        draw_thot(overlayed_image, thot_rect)
    
    return overlayed_image

def main_file(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print("Unable to read image [ {} ]".format(file_path))
        return
    
    overlay = thot_overlay(img)

    cv2.imwrite('output.png', overlay)
    print("Output to output.png")

def main_camera():
    camera_capture = cv2.VideoCapture(0)
    if camera_capture is None or not camera_capture.isOpened():
        print("Unable to open camera.")
        return
    
    while True:
        ret, img = camera_capture.read()
        overlay = thot_overlay(img)
        cv2.imshow('Thot Detector', overlay)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    camera_capture.release()
    cv2.destroyAllWindows()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) == 3 and argv[1] == "file":
        main_file(argv[2])
    elif len(argv) == 2 and argv[1] == "camera":
        main_camera()
    else:
        print("Usage: python main.py <file [path]|camera>")