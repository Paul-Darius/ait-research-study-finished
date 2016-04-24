import os
import sys
import random
import shutil
from collections import defaultdict


train = open("Caffe_Files/train.txt",'r')
train1 = open("Caffe_Files/train1.txt",'w+')
train2 = open("Caffe_Files/train2.txt",'w+')

print "Wait a second. A big database is being generated, the process may take few minutes!"

labeled = defaultdict(list)

while 1:
    txt = train.readline()
    if txt == '':
        break
    source,_,label = txt.rpartition(' ')
    labeled[int(label)].append(source)


for k in range(0, len(labeled)-1):
    for i in range(0, len(labeled[k])-1):
        for j in range(i+1, len(labeled[k])-1):
            train1.write(labeled[k][i]+' 1\n')
            train2.write(labeled[k][j]+' 1\n')
            train1.write(labeled[k][i]+' 0\n')
            while True:
                rd = random.choice(range(1,i-1) + range(i+1,len(labeled)-1))
                if labeled[rd] != []:
                    break
            train2.write(labeled[rd][random.randint(0,len(labeled[rd])-1)]+' 0\n')

train.close()
train1.close()
train2.close()

# Same for the test file.

train = open("Caffe_Files/test.txt",'r')
train1 = open("Caffe_Files/test1.txt",'w+')
train2 = open("Caffe_Files/test2.txt",'w+')

labeled = defaultdict(list)

while 1:
    txt = train.readline()
    if txt == '':
        break
    source,_,label = txt.rpartition(' ')
    labeled[int(label)].append(source)


for k in range(0, len(labeled)-1):
    for i in range(0, len(labeled[k])-1):
        for j in range(i+1, len(labeled[k])-1):
            train1.write(labeled[k][i]+' 1\n')
            train2.write(labeled[k][j]+' 1\n')
            train1.write(labeled[k][i]+' 0\n')
            while True:
                rd = random.choice(range(1,i-1) + range(i+1,len(labeled)-1))
                if labeled[rd] != []:
                    break
            train2.write(labeled[rd][random.randint(0,len(labeled[rd])-1)]+' 0\n')

train.close()
train1.close()
train2.close()
