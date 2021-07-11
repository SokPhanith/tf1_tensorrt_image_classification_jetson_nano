import argparse
import os
def get_parser():
  parser = argparse.ArgumentParser(description="Runtime with Tensorflow1 Image_Classification")
  parser.add_argument("--model",default="model.pb",help="path to frozen graph model.")
  parser.add_argument("--input_name",default="input",help="name of input layer of model default[input].")
  parser.add_argument("--output_name",default="MobilenetV2/Predictions/Reshape_1",help="name of output layer of model default[MobilenetV2/Predictions/Reshape_1].")
  parser.add_argument("--label",default="labels.txt",help="path to labels.txt one line label.")
  parser.add_argument("--csi", action="store_true",help="set --csi for using pi-camera-v2")
  parser.add_argument("--cpu", action="store_true",help="set --cpu for inferance with cpu default[GPU]")
  parser.add_argument("--mean", action="store_true",help="set --mean for Resnetv1 and vgg_net")
  parser.add_argument("--webcam", type=str, default=None,help="Take inputs from webcam /dev/video*.")
  parser.add_argument('--image', type=str, default=None,help='path to image file name')
  parser.add_argument("--video",type=str,default=None,help="Path to video file.")
  parser.add_argument("--height",type=int,default=224,help="height input image default[224].")
  parser.add_argument("--width",type=int,default=224,help="width input image default[224].")
  parser.add_argument("--height_display",type=int,default=480,help="height display image default[480].")
  parser.add_argument("--width_display",type=int,default=640,help="width display image default[640].")
  parser.add_argument("--batch_size",type=int,default=1,help="batch size input image default[1].")
  parser.add_argument("--channel",type=int,default=3,help="channel input image default[3].")
  return parser
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
args = get_parser().parse_args()
if args.cpu:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  print('run with cpu')
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
  print('run with gpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2 
import time
means = np.array([[[123.68, 116.779, 103.939]]], dtype=np.float32) #vgg_net and resnetv1
if args.csi:
  print("csi using")
  cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0,display_width=args.width_display,display_height=args.height_display),cv2.CAP_GSTREAMER)
elif args.image:
  print("image for classification")
  print(args.image)
elif args.webcam:
  print('webcam using')
  cam = open_cam_usb(int(args.webcam),args.width_display,args.height_display,USB_GSTREAMER=True)
elif args.video:
  print('video for classification')
  cam = cv2.VideoCapture(args.video)
else:
  print('None source for input need image, video, csi or webcam')
  sys.exit()
print('Loading label : ',args.label)
with open(args.label) as labels:
  classes = [i.strip() for i in labels.readlines()]
if ":" in classes[0]:
  labels = []
  for i in classes:
    labels.append(i.split(":")[1])
else:
  labels = classes
print('Start loading model : ',args.model)
gd = tf.compat.v1.GraphDef.FromString(open(args.model, 'rb').read())
inp, predictions = tf.import_graph_def(gd,return_elements = [args.input_name+':0', args.output_name+':0'])
print('loading done')
with tf.compat.v1.Session(graph=inp.graph):
  while True:
    if args.image:
      frame = cv2.imread(args.image)
    else:
      _, frame = cam.read()
    img = cv2.resize(frame,(args.width,args.height))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if args.mean:
      img = img.astype(np.float32) - means
    else:
      img = (img.astype(np.float32)/127.5) - 1.0
    t1 = time.time()
    x = predictions.eval(feed_dict={inp: img.reshape(args.batch_size,args.width,args.height,args.channel)})
    classes = labels[x.argmax()]
    percent = float(x.max())
    dt = time.time() - t1
    cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (11, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, str(round(1.0/dt,2))+' fps', (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(frame, str(round(1.0/dt,2))+' fps', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 240, 240), 1, cv2.LINE_AA)   
    cv2.imshow('Demo',frame)
    cv2.moveWindow('Demo',0,0)
    if cv2.waitKey(1) == ord('q'):
      break
  if args.image:
    cv2.destroyAllWindows()
  else:
    cv2.destroyAllWindows()
    cam.release()



