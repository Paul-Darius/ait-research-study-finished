import os
import sys
import random
import shutil
from collections import defaultdict
import scipy.io as sio


full_db = open("Caffe_Files/full_database.txt",'r')

print "Wait a second. A big database is being generated, the process may take few minutes!"

labeled = defaultdict(list)

line_number=1
while 1:
    txt = full_db.readline()
    if txt == '':
        break
    source,_,label = txt.rpartition(' ')
    labeled[int(label)].append(line_number)
    line_number+=1

pos_pairs = []
neg_pairs = []

total_length_pairs = 0

for k in range(0, len(labeled)-1):
    if total_length_pairs >= 3000:
        break
    for i in range(0, len(labeled[k])-1):
        for j in range(i+1, len(labeled[k])-1):
            pos_pairs.append([labeled[k][i],labeled[k][j]])
            while True:
                rd = random.choice(range(1,i-1) + range(i+1,len(labeled)-1))
                if labeled[rd] != []:
                    break
            neg_pairs.append([labeled[k][i],labeled[rd][random.randint(0,len(labeled[rd])-1)]])
            total_length_pairs+=1

dic = {'pos_pair':pos_pairs, 'neg_pair':neg_pairs}

sio.savemat("src/face_id/face_verification_experiment-master/code/mbk_pair.mat", dic)

full_db.close()
