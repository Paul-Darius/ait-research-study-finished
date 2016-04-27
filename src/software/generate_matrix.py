from math import factorial as fac, pow
import operator
import matplotlib.pyplot as plt

def binomial(x, y):
    try:
        binom = fac(y) // fac(x) // fac(y - x)
    except ValueError:
        binom = 0
    return binom

def generate_matrix(tpr,fpr,frame_number, picture_number):
	p = tpr/(tpr+fpr)
	print p 
	#### PPD = matrix of Probability to be Positive knowing that it has been Detected. PPND = P P knowing that it is Not Detected
	PPD = [[0 for x in range(picture_number+1)] for y in range(frame_number+1)]
	PPND = [[0 for x in range(picture_number+1)] for y in range(frame_number+1)]
	for i in range(0,picture_number+1):
		a_i=0
		b_i=0
		for i2 in range(i,picture_number+1):
			a_i+=binomial(i2,picture_number)*pow(p,i2)*pow(1-p,picture_number-i2)
		b_i = 1-a_i
		for j in range(0, frame_number+1):
			if i != 0 and j != 0:
				for j2 in range(j,frame_number+1):
					PPD[j][i]+=binomial(j2,frame_number)*pow(a_i,j2)*pow(1-a_i,frame_number-j2)
				for j3 in range(frame_number-j+1,frame_number+1):
					PPND[j][i]+=binomial(j3,frame_number)*pow(b_i,j3)*pow(1-b_i,frame_number-j3)
	return PPD,PPND

def extract_max(double_list):
	index_frame, value = max(enumerate(double_list), key=operator.itemgetter(1))
	index_picture, value = max(enumerate(value), key=operator.itemgetter(1))
	return index_frame, index_picture, value


###### How to use this:	
PPD,PPND=generate_matrix(0.88,0.11,5,3)
print PPD, PPND
#f,p,v = extract_max(TPR)
#print "Optimal case is : You should consider that with at least "+str(f)+" detected faces detected as valid on at least "+str(p)+" picture(s) of the criminal you are looking for you have a "+str(v)+" true positive rate."
#print "It corresponds to a "+str(FPR[f][p])+" false positive rate"
im = plt.imshow(PPD, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()

im = plt.imshow(PPND, cmap = 'hot')
plt.colorbar(im, orientation='horizontal')
plt.show()
