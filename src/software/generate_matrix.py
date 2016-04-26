from math import factorial as fac, pow
import operator


def binomial(x, y):
    try:
        binom = fac(y) // fac(x) // fac(y - x)
    except ValueError:
        binom = 0
    return binom

def generate_matrix(tpr,fpr,frame_number,picture_number): 
	TPR = [[0 for x in range(picture_number+1)] for y in range(frame_number+1)]
	FPR = [[0 for x in range(picture_number+1)] for y in range(frame_number+1)]
	for i in range(0,picture_number+1):
		a_i=0
		b_i=0
		for i2 in range(i,picture_number+1):
			a_i+=binomial(i2,picture_number)*pow(tpr,i2)*pow(1-tpr,picture_number-i2)
			b_i += binomial(i2,picture_number)*pow(fpr,i2)*pow(1-fpr,picture_number-i2)
		for j in range(0, frame_number+1):
			if i != 0 and j != 0:
				TPR[j][i]=binomial(j,frame_number)*pow(a_i,j)*pow(1-a_i,frame_number-j)
				FPR[j][i]=binomial(j,frame_number)*pow(b_i,j)*pow(1-b_i,frame_number-j)
	return TPR,FPR

def extract_max(double_list):
	index_frame, value = max(enumerate(double_list), key=operator.itemgetter(1))
	index_picture, value = max(enumerate(value), key=operator.itemgetter(1))
	return index_frame, index_picture, value


###### How to use this:	
#TPR,FPR=generate_matrix(0.88,0.11,5,3)
#f,p,v = extract_max(TPR)
#print "Optimal case is : You should consider that with at least "+str(f)+" detected faces detected as valid on at least "+str(p)+" picture(s) of the criminal you are looking for you have a "+str(v)+" true positive rate."
#print "It corresponds to a "+str(FPR[f][p])+" false positive rate"