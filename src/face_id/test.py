import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

# Network
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('face_verification_experiment-master/proto/LightenedCNN_A_deploy.prototxt', 'face_verification_experiment-master/model/LightenedCNN_A.caffemodel', caffe.TEST);

im = np.array(Image.open('picoo.jpg'))
im = im.reshape(1,128,128)

net.blobs['data'].data[...] = im
out = net.forward()
print "DATA IS ARRIVING"
#for l in net.params['conv1'][1]:
#	print l
a = [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]


for l in a:
	print l
