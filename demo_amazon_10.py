import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

#classes_name =  ["bakingpan", "colander", "cup", "pitcher", "saucepan", "scissors", "shaker", "thermos"]
classes_name =  ["dvd","socks", "gum", "salt", "icetray", "notebook", "wipe", "joke", "mousetrap", "colorpencil"]

def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:10]
  C = predicts[0, :, :, 10:12]
  coordinate = predicts[0, :, :, 12:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 10))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes
  print "here is P"
  print P[5,1, 0, :]
  print P
  index = np.argmax(P)
  print(index)
  #index=752
  index = np.unravel_index(index, P.shape)
  print(index)
  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))
  print "coordinate is:"
  print(coordinate)
  max_coordinate = coordinate[index[0], index[1], index[2], :]
  print "maximum coordinate is:"
  print(max_coordinate)
  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num

common_params = {'image_size': 448, 'num_classes': 10, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

np_img = cv2.imread('amazontest10/pic35.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, 'models/train/amazon10.ckpt-57500')

np_predict = sess.run(predicts, feed_dict={image: np_img})

xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),(255, 255, 255),5)
cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (255, 255, 255))
cv2.imwrite('0021_out.jpg', resized_img)
sess.close()
