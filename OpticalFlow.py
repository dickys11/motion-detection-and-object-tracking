# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import imutils
import cv2
import sys


# Global vars:
WIDTH = 700
STEP = 16
QUIVER = (255, 100, 0)


def draw_flow(img, flow, step=STEP):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = cv2.VideoCapture(fn)
    ret, prev = cam.read()
    prev = imutils.resize(prev, width=WIDTH)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    ret, frame1 = cam.read()
    frame1 = imutils.resize(frame1, width=WIDTH)
    ret, frame2 = cam.read()
    frame2 = imutils.resize(frame2, width=WIDTH)

    while True:
        ret, img = cam.read()
        img = imutils.resize(img, width=WIDTH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        cv2.imshow('flow', draw_flow(gray, flow))

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("contours", frame1)
        frame1 = frame2
        ret, frame2 = cam.read()
        frame2 = imutils.resize(frame2, width=WIDTH)

        ch = cv2.waitKey(40)
        if ch == 27:
            break
    cv2.destroyAllWindows()
