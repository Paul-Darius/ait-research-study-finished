import cv2
import sys
import numpy as np
from PIL import Image
import caffe
from collections import OrderedDict
import math
from math import fabs as abs
import shutil
import os
import dlib
from skimage import io
from PIL import Image
import generate_matrix
import re
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
address_of_images_of_criminal.append("src/software/criminal_pictures/1.jpg")
address_of_images_of_criminal.append("src/software/criminal_pictures/2.jpg")
address_of_images_of_criminal.append("src/software/criminal_pictures/3.jpg")

########### THEN I HAVE TO COMPUTE THE FEATURES OF THESE IMAGES

###### I DEFINE A FUNCTION EXTRACTED FROM CAFE_FTR.PY WHICH WILL RETURN THE ARRAY OF FEATURES CORRESPONDING TO  ADDRESS_OF_IMAGES_OF_CRIMINALS

#network_proto_path, network_model_path = network_path
net = caffe.Classifier('src/face_id/face_verification_experiment-master/proto/LightenedCNN_A_deploy.prototxt', 'src/face_id/face_verification_experiment-master/model/LightenedCNN_A.caffemodel')
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

if not os.path.exists("Captured_Faces"):
    os.makedirs("Captured_Faces")
first_level_in_database = []

for (dirpath, dirnames, filenames) in os.walk("Database"):
    first_level_in_database.extend(dirnames)
    break
first_level_in_database=sorted(first_level_in_database)

####### MAIN LOOP STARTS HERE


#### FOR ALL THE FOLDERS CONTAINING DETECTED FACES IN THE VIDEO
for count in range(len(first_level_in_database)):
    filenames = []
    for (_, _, filename) in os.walk("Database/"+first_level_in_database[count]):
        filenames.append(filename)
        break
    filenames = filenames[0]
    for i in range(0,len(filenames)):
        filenames[i]="Database/"+first_level_in_database[count]+'/'+filenames[i]
    if not os.path.exists("Captured_Faces/"+first_level_in_database[count]):
        os.makedirs("Captured_Faces/"+first_level_in_database[count])

    ##### FOR EACH FACE OF THE VIDEO
    for i in range(len(filenames)):
        current_label_names=[]
        try:
            current_label = re.search('Label(.+?)Frame', filenames[i]).group(1)
        except AttributeError:
            current_label = '0'
        current_label=int(current_label)
        ###### WE FIND ALL THE OTHER FACES WITH THE SAME LABEL AND PUT IT IN current_label_names
        for j in range(len(filenames)):
            if filenames[j].rsplit('/',1)[-1][0]=='L' or filenames[j].rsplit('/',1)[-1][0]=='L':
                label = int(re.search('Label(.+?)Frame', filenames[j]).group(1))
                if label == current_label:
                    current_label_names.app ooend(filenames[j])
        #### WE APPLY THE STRATEGY OF SELECTING THE TEST WHICH GIVES THE BEST PROBABILITY OF MAKING THE RIGHT DECISION
        p_pd,p_pnd = generate_matrix.generate_matrix(0.88,0.12,len(current_label_names),len(address_of_images_of_criminal))
        min_number_suspects, min_number_criminal, _ = generate_matrix.extract_max(p_pd,p_pnd,len(current_label_names),len(address_of_images_of_criminal))
        actual_smallest_number_validated_suspects = 0
        actual_smallest_number_validated_criminal_pictures = 0
        for y in range(0,len(current_label_names)):
            feats = extract_feature(net, [current_label_names[y]], 'prob', 1)
            feats = feats[0]
            actual_smallest_number_validated_criminal_pictures = 0
            for z in range(0,len(address_of_images_of_criminal)):
                if cosine_similarity(feats, features_criminal[z]) > 0.2:
                    actual_smallest_number_validated_criminal_pictures+=1
            if actual_smallest_number_validated_criminal_pictures >= min_number_criminal:
                actual_smallest_number_validated_suspects+=1
        if actual_smallest_number_validated_suspects>=min_number_suspects or 1:
            for k in range(len(current_label_names)):
                #if current_label_names[k].rsplit('/',1)[-1] not in os.walk("tmp"):
                print "hello"
                shutil.copyfile(current_label_names[k],"Captured_Faces/"+first_level_in_database[count]+'/'+current_label_names[k].rsplit('/',1)[-1])