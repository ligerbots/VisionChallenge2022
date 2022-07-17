#!/usr/bin/env python3

import sys
import math
import cv2
import numpy                    # might be needed
import matplotlib.pyplot as plt


class CrossFinder:
    # height of camera off the floor
    CAMERA_HEIGHT = 33.0   # inches

    # vertical tilt of the camera
    CAMERA_ANGLE = math.radians(31)

    # camera field of view
    # Assumes that the camera is turned 90 degrees, like in 2022
    CAMERA_HFOV = math.radians(50.0)
    CAMERA_VFOV = math.radians(79.5)

    # height of the center of the target off the floor
    TARGET_HEIGHT_FROM_FLOOR = 72.0   # inches

    # target dimensions (inches)
    TARGET_WIDTH = 8.5
    TARGET_HEIGHT = 11.0

    def __init__(self):
        '''Initialization routine
           put any instance variable initialization here'''

        self.target_center = None
        self.target_contour = None
        return

    def process_image(self, camera_frame):
        '''Main image processing routine
        camera_frame is a valid OpenCV image frame in BGR format

        return: should return a list of values:
             success, distance, angle
        where "success" = True if found, False if could not find good cross
        angle should be in degrees'''

        # clear values from previous image
        self.target_center = self.target_contour = None

        img = camera_frame

        threshold = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        myH, myS, myV = 79, 224, 208

        hsvThreshold = 60

        # plt.imshow(img)
        filtered = cv2.inRange(threshold, (myH - hsvThreshold, myS - hsvThreshold, myV - hsvThreshold), (myH + hsvThreshold, myS + hsvThreshold, myV + hsvThreshold))

        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            return False, 0.0, 0.0

        hull = contours[0]
        maxLen = len(hull)

        self.target_contour = contours[0]

        for contour in contours:
            temp = cv2.convexHull(contour)
            if len(temp) > maxLen:
                hull = temp
                maxLen = len(temp)
                self.target_contour = contour

        epsilon = 0.01* cv2.arcLength(hull, True)
        hull_approx = cv2.approxPolyDP(hull, epsilon, True)

        left = right = hull_approx[0][0][0]
        top = bot = hull_approx[0][0][1]

        for pt in hull_approx:
            left = min(left, pt[0][0])
            right = max(right, pt[0][0])
            top = min(top, pt[0][1])
            bot = max(bot, pt[0][1])

        # cv2.drawContours(img, [hull_approx], -1, (255, 0, 0), 2)
        # cv2.drawContours(img, targetContour, -1, (255, 0, 0), 2)

        # cv2.circle(img, (left, top), 8, (255, 0, 0), -1)
        # cv2.circle(img, (right, top), 6, (0, 255, 0), -1)
        # cv2.circle(img, (left, bot), 4, (0, 0, 255), -1)
        # cv2.circle(img, (right, bot), 2, (255, 255, 255), -1)

        hull_area = cv2.contourArea(hull_approx)
        target_width = right - left
        target_height = bot - top

        target_height_aspect_ratio = CrossFinder.TARGET_HEIGHT/float(target_height)
        target_width_aspect_ratio = CrossFinder.TARGET_WIDTH/float(target_width)
        target_area_ratio = (CrossFinder.TARGET_HEIGHT * CrossFinder.TARGET_WIDTH)/float(target_width * target_height)

        print(target_height_aspect_ratio, target_width_aspect_ratio, target_area_ratio)

        # comparing the ratio of side lengths of the target
        if not abs(target_width_aspect_ratio - target_height_aspect_ratio) <= 0.15:
            return False, 0.0, 0.0

        # comparing the ratio of area of the target
        if not abs(target_area_ratio - target_height_aspect_ratio*target_height_aspect_ratio) <= 0.3:
            return False, 0.0, 0.0

        self.target_center = [(left+right)/2, (top+bot)/2]

        # plt.imshow(img)
        # plt.show()

        return True, 0.0, 0.0

    def prepare_output_image(self, camera_frame):
        '''Prepare output image for drive station.
        Add any mark up to the output frame as if you are sending it to the Drivers Station.
        For debugging, it is helpful to put extra stuff on the image.'''

        # make a copy of the original to be marked up
        output_frame = camera_frame.copy()

        # add any markup here. Look at OpenCV routines like (examples):
        #   drawMarker(), drawContours(), text()

        # print(self.target_center[0], self.target_center[1])
        # example: put a red cross at location 200, 200
        if self.target_center is not None:
            cv2.drawMarker(output_frame, (int(self.target_center[0]), int(self.target_center[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)

        return output_frame


# --------------------------------------------------------------------------------
# Main routines, used for running the finder by itself for debugging and timing

def process_files(finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path
    import re

    print('File,Success,Distance,Angle')

    for image_file in input_files:
        bgr_frame = cv2.imread(image_file)

        success, distance, angle = finder.process_image(bgr_frame)
        print(image_file, success, round(distance, 1), round(angle, 1), sep=',')

        bgr_frame = finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))

        # output as PNG, because that does not do compression
        outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
        cv2.imwrite(outfile, bgr_frame)

    return


def time_processing(finder, input_files):
    '''Time the processing of the test files'''

    from time import time

    startt = time()

    cnt = 0
    proct = 0

    # Loop 100x over the files. This is needed to make it long enough
    #  to get reasonable statistics. If we have 100s of files, we could reduce this.
    # Need the total time to be many seconds so that the timing resolution is good.

    for _ in range(100):
        for image_file in input_files:
            bgr_frame = cv2.imread(image_file)

            pst = time()
            finder.process_image(bgr_frame)
            proct += time() - pst

            cnt += 1

    deltat = time() - startt

    print('Net:    {0} frames in {1:.3f} seconds = {2:.2f} ms/call, {3:.2f} FPS'.format(cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    print('Finder: {0} frames in {1:.3f} seconds = {2:.2f} ms/call, {3:.2f} FPS'.format(cnt, proct, 1000.0 * proct / cnt, cnt / proct))
    return


def main():
    '''Main routine for testing a Finder'''
    import argparse

    parser = argparse.ArgumentParser(description='finder test routine')
    parser.add_argument('--output-dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    finder = CrossFinder()

    if sys.platform == "win32":
        # windows does not expand the "*" files on the command line
        #  so we have to do it.
        import glob

        infiles = []
        for f in args.input_files:
            infiles.extend(glob.glob(f))
        args.input_files = infiles

    if args.output_dir is not None:
        process_files(finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(finder, args.input_files)

    return


if __name__ == '__main__':
    main()
