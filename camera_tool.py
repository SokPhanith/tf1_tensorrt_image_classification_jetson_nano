import os 
import cv2 
import time
import argparse
from random import randint
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0):
    return ("nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=True"
            %(capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,))
def open_cam_usb(dev,width,height,USB_GSTREAMER):
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)
def get_parser():
    parser = argparse.ArgumentParser(description="TensorRT Runtime with Tensorflow1 Image_classification")
    parser.add_argument("--dataset_dir",default="dataset",help="path to make folder dataset dir.")
    parser.add_argument("--label",default="classes.txt",help="path to classes.txt one line label and no-background.")
    parser.add_argument("--csi", action="store_true",help="set --csi for using pi-camera")
    parser.add_argument("--webcam", type=str, default=None,help="Take inputs from webcam /dev/video*.")
    parser.add_argument("--height_display",type=int,default=480,help="height display and save image default[480].")
    parser.add_argument("--width_display",type=int,default=640,help="width display and save image default[640].")
    return parser
if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.csi:
        print("csi using")
        cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0,display_width=args.width_display,display_height=args.height_display),cv2.CAP_GSTREAMER)
    elif args.webcam:
        print('webcam using')
        cam = open_cam_usb(int(args.webcam),args.width_display,args.height_display,USB_GSTREAMER=True)
    else:
        print('None source for take image, need csi or webcam')
        sys.exit()
    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)
    print('\t')
    print('Start taking image for training dataset:')
    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)
    with open(args.label) as labels:
        classes = [i.strip() for i in labels.readlines()]
    for single_label in classes:
        if not os.path.exists(args.dataset_dir+'/'+single_label):
            os.makedirs(args.dataset_dir+'/'+single_label)
    count = 0
    image_name = randint(10,100)
    print('taking image for class : ',classes[count])
    while True:
        _, frame = cam.read()
        cv2.imshow('q for exit - s for save - c for change class',frame)
        cv2.moveWindow('q for exit - s for save - c for change class',0,0)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(args.dataset_dir+'/'+classes[count]+'/'+str(image_name)+'.jpg',frame)
            print(args.dataset_dir+'/'+classes[count]+'/'+str(image_name)+'.jpg')
            image_name += 1
        if key == ord('c'):
            if count == len(classes) - 1:
                count = 0
                print('taking image for class : ',classes[count])
            else:
                count += 1
                print('taking image for class : ',classes[count])
    cv2.destroyAllWindows()
    cam.release()

