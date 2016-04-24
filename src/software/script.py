import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from collections import OrderedDict
import math
import os
import dlib
import glob
from skimage import io
from pymatbridge import Matlab
from PIL import Image

############ HERE I ASK FOR THE IMAGES AND I PREPROCESS THEM
######## THE FINAL RESULT IS CONTAINED IN THE ARRAY ADDRESS_OF_IMAGES_OF_CRIMINAL

address_of_images_of_criminal = []

##### THESE LINES WILL BE USED IN FINAL VERSION
#print "How many pictures of the criminal do you want to provide?"
#number_pictures = int(raw_input())
#for i in range(0, number_pictures):
#	print "Address of image number "+str(i+1)
#	address_of_images_of_criminal.append(raw_input())
#####

##### AT THE MOMENT WE USE THESE IMAGES FOR TESTING
address_of_images_of_criminal.append("criminal_pictures/1.jpg")
address_of_images_of_criminal.append("criminal_pictures/2.jpg")
address_of_images_of_criminal.append("criminal_pictures/3.jpg")

########### THEN I HAVE TO COMPUTE THE FEATURES OF THESE IMAGES

###### I DEFINE A FUNCTION EXTRACTED FROM CAFE_FTR.PY WHICH WILL RETURN THE ARRAY OF FEATURES CORRESPONDING TO  ADDRESS_OF_IMAGES_OF_CRIMINALS

#network_proto_path, network_model_path = network_path
net = caffe.Classifier('../face_id/face_verification_experiment-master/proto/LightenedCNN_A_deploy.prototxt', '../face_id/face_verification_experiment-master/model/LightenedCNN_A.caffemodel')
caffe.set_mode_cpu()

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
    
    #if data_mean is None:
        #data_mean = np.zeros(1)
    #net.transformer.set_mean('data', data_mean)
    #if not image_as_grey:
    #    net.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    #net.set_input_scale('data', 256)  # the reference model operates on images in [0,255] range instead of [0,1]
net.transformer.set_input_scale('data', 1)

    #img_list = [caffe.io.load_image(p) for p in image_file_list]

    #----- test

def extract_feature(net, image_list, layer_name, image_as_grey = False):
    """
    Extracts features for given model and image list.

    Input
    network_proto_path: network definition file, in prototxt format.
    network_model_path: trainded network model file
    image_list: A list contains paths of all images, which will be fed into the
                network and their features would be saved.
    layer_name: The name of layer whose output would be extracted.
    save_path: The file path of extracted features to be saved.
    """
    
    blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])

    #blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
    shp = blobs[layer_name].shape
    #print blobs['data'].shape

    batch_size = blobs['data'].shape[0]
    #print blobs[layer_name].shape
    #print 'debug-------\nexit'
    #exit()

    #params = OrderedDict( [(k, (v[0].data,v[1].data)) for k, v in net.params.items()])
    if len(shp) is 2:
        features_shape = (len(image_list), shp[1])
    elif len(shp) is 3:
        features_shape = (len(image_list), shp[1], shp[2])
    elif len(shp) is 4:
        features_shape = (len(image_list), shp[1], shp[2], shp[3])

    features = np.empty(features_shape, dtype='float32', order='C')
    img_batch = []
    for cnt, path in zip(range(features_shape[0]), image_list):
        img = caffe.io.load_image(path, color = not image_as_grey)
        if image_as_grey and img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]
        #if cnt == 0:
        #    print 'image shape: ', img.shape
        #print img[0:10,0:10,:]
        #exit()
        img_batch.append(img)
        #print 'image shape: ', img.shape
        #print path, type(img), img.mean()
        if (len(img_batch) == batch_size) or cnt==features_shape[0]-1:
            scores = net.predict(img_batch, oversample=False)
            '''
            print 'blobs[%s].shape' % (layer_name,)
            tmp =  blobs[layer_name]
            print tmp.shape, type(tmp)
            tmp2 = tmp.copy()
            print tmp2.shape, type(tmp2)
            print blobs[layer_name].copy().shape
            print cnt, len(img_batch)
            print batch_size
            #exit()

            #print img_batch[0:10]
            #print blobs[layer_name][:,:,0,0]
            #exit()
            '''

            # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
            # syncs the memory between GPU and CPU
            blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])

            #print '%d images processed' % (cnt+1,)

            #print blobs[layer_name][0,:,:,:]
            # items of blobs are references, must make copy!
            features[cnt-len(img_batch)+1:cnt+1] = blobs[layer_name][0:len(img_batch)].copy()
            img_batch = []

        #features.append(blobs[layer_name][0,:,:,:].copy())
    features = np.asarray(features, dtype='float32')
    return features

features_criminal = extract_feature(net, address_of_images_of_criminal, 'prob', 1)

##### THIS FUNCTION WILL BE NEEDED FOR SOME REASON

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

########## NOW WE WORK ON THE FACE DETECTION
cascPath = "../others/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

##### WE HAVE TO PREPROCESS THE DETECTED FACE BEFORE APPLYING OUR FILTER.

#### THOSE LINES DEFINE THE PREDICTOR FOR THE KEY POINTS ON EACH FACE
predictor_path = "../others/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


####### SOME FUNCTIONS HAVE TO BE DEFINED FIRST


def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image


######## NOW THE PREPROCESS FUNCTION

def preprocess(string):
    """
    This function takes a gray face as input and return the preprocessed face as output.
    Pre-processing is in 3 steps: RGB to gray (already done), resize to 144x144,
    and apply the image to a matlab script which will do some other modifications

    """
    image = io.imread(string)
    dets = detector(image, 1)
    for k, d in enumerate(dets):
        shape = predictor(image, d)
        left_eye = shape.part(36)
        right_eye = shape.part(45)
        left_eye = [left_eye.x,left_eye.y]
        right_eye = [right_eye.x,right_eye.y]  
        img = Image.open("face.png")          
        CropFace(img, eye_left=left_eye, eye_right=right_eye, offset_pct=(0.20,0.20), dest_sz=(144,144)).save(string)

###### THOSE LINES WILL BE USED IN THE FINAL VERSION
#print "What is the address of the video to be analyzed?"
#address = raw_input()
address="video/video2.avi"
video_capture = cv2.VideoCapture(address)
cv2.namedWindow('Video',cv2.WINDOW_NORMAL)


###### WE HAVE TO PREPROCESS THE IMAGES OF THE CRIMINAL
for i in range(0,len(address_of_images_of_criminal)-1):
  preprocess(address_of_images_of_criminal[i])



#### THIS LINES ARE JUST FOR THE DEMO
for i in range(0,330):
	ret, frame = video_capture.read()

criminal_face_number = 0
while(video_capture.isOpened()):
	# Capture frame-by-frame
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	print "**** NEW FRAME ****"
# Draw a rectangle around the faces
	i=0
	for (x, y, w, h) in faces:
		roi = frame[y:y+h, x:x+w]
		### HERE MANY THINGS WILL BE MODIFIED
		cv2.imwrite("face.png", roi)
		feats = extract_feature(net, ["face.png"], 'prob', 1)
		feats = feats[0]
		preprocess("face.png")
		print "**NEW_FACE**"
		for i in range(0,len(features_criminal)):
			is_criminal=0
			print cosine_similarity(feats,features_criminal[i])
			if (cosine_similarity(feats,features_criminal[i])>0.1308):
				is_criminal = 1
			if is_criminal==0:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			else:
				print "CRIMINAL DETECTED"
				roi = frame[y:y+h, x:x+w]
				cv2.imwrite("detected/criminal"+str(criminal_face_number)+"Score"+str(cosine_similarity(feats,features_criminal[i]))+".png", roi)
				criminal_face_number+=1
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame,'CRIMINAL DETECTED',(x,y), font, 2,(0,0,255))
		os.remove("face.png")
		i+=1

	# Display the resulting frame
	cv2.namedWindow('Video', cv2.WINDOW_NORMAL) 
	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
mlab.stop()