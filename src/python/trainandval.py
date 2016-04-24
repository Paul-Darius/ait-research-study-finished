import os
import sys
import random
import shutil

number_of_input_per_label = dict() # is a dictionnary

full_db = open("Caffe_Files/full_database.txt", "r")

while 1:
	txt = full_db.readline()
	if txt =='':
		break
	source,_,key = txt.rpartition(' ')
	key=int(key)
	if key in number_of_input_per_label:
		number_of_input_per_label[key] += 1
	else:
		number_of_input_per_label[key] = 1

full_db.close()

# Random permutation of the lines of the file.

with open("Caffe_Files/full_database.txt", "r") as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('Caffe_Files/full_database_tmpcpy.txt','w+') as target:
    for _, line in data:
        target.write( line )
# Now we can generate train.txt, valid.txt and test.txt using the permutated full database.

if (len(sys.argv)==1 or (len(sys.argv)==2 and sys.argv[1]=="3")):
	full_db_permutated = open("Caffe_Files/full_database_tmpcpy.txt", "r")
	train = open('Caffe_Files/train.txt','w+')
	valid = open('Caffe_Files/valid.txt','w+')
	test = open('Caffe_Files/test.txt','w+')

	train_label_number = dict() # is a dictionnary
	valid_label_number = dict() # is a dictionnary
	test_label_number = dict() # is a dictionnary

	while 1:
		txt = full_db_permutated.readline()
		if txt =='':
			break
		source,_,key = txt.rpartition(' ')
		key=int(key)
		if key in train_label_number:
			if number_of_input_per_label[key]/3 >= train_label_number[key]:
				train_label_number[key] += 1
				train.write(txt) # copy txt in train.txt
			else:
				if key in valid_label_number:
					if number_of_input_per_label[key]/3 >= valid_label_number[key]:
						valid_label_number[key] += 1
						valid.write(txt) # copy txt in valid.txt
					else:
						test.write(txt) # copy txt in test.txt
						if key in test_label_number:
							test_label_number[key] += 1
						else:
								test_label_number[key] = 1
				else:
					valid_label_number[key] = 1
					valid.write(txt)
		
		else:
			train_label_number[key] = 1
			train.write(txt)
	
	full_db_permutated.close()
	os.remove('Caffe_Files/full_database_tmpcpy.txt')
	train.close()
	valid.close()
	test.close()
elif (len(sys.argv)==2 and sys.argv[1]=="2"): # Second scenario: the argument is 2 ie two files created.
	full_db_permutated = open("Caffe_Files/full_database_tmpcpy.txt", "r")
	train = open('Caffe_Files/train.txt','w+')
	test = open('Caffe_Files/test.txt','w+')

	train_label_number = dict() # is a dictionnary
	test_label_number = dict() # is a dictionnary

	while 1:
		txt = full_db_permutated.readline()
		if txt =='':
			break
		source,_,key = txt.rpartition(' ')
		key=int(key)
		if key in train_label_number:
			if number_of_input_per_label[key]/2 >= train_label_number[key]:
				train_label_number[key] += 1
				train.write(txt) # copy txt in train.txt
			else:
				test.write(txt) # copy txt in test.txt
				if key in test_label_number:
					test_label_number[key] += 1
				else:
					test_label_number[key] = 1
		else:
			train_label_number[key] = 1
			train.write(txt)
	full_db_permutated.close()
	os.remove('Caffe_Files/full_database_tmpcpy.txt')

	train.close()
	test.close()
elif (len(sys.argv)==2 and sys.argv[1]=="1"):
	full_db_permutated = open("Caffe_Files/full_database_tmpcpy.txt", "r")
	train = open('Caffe_Files/train.txt','w+')
	shutil.copyfileobj(full_db_permutated, train)
	full_db_permutated.close()
	os.remove('Caffe_Files/full_database_tmpcpy.txt')
	train.close()

else:
	print "Error on the arguments of this script. See the README file for more informations."