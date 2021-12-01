import mir_eval
import sklearn#.metrics.confusion_matrix
import sklearn.metrics
import numpy as np
import random
def convert_onehots_to_num(one_hots):
	pitches = []
	for i in one_hots:
		j = np.array(i)
		x = np.where(j > 0)
		pitches.append(x[0])
	#print(pitches)
	return pitches

def convertVarToHz(i):
	midi =  (i/721)*(83-38) + 38
	hz = 2**((midi-69)/12)*440
	if i == 722:
		return 0;
	else:
		return hz

def convertArrayToCents(arr):
	hzVals = np.array([convertVarToHz(xi) for xi in arr])
	#print(arr[2])
	hzVals = hzVals.reshape(-1)
	return mir_eval.melody.hz2cents(hzVals)
def if722(i):
	if i == 722:
		return 0
	else:
		return 1
def getVoicedArray(arr):
	hzVals = np.array([if722(xi) for xi in arr])
	hzVals = hzVals.reshape(-1)
	return hzVals


def rca(y_pred, y_test):
	y_p = convert_onehots_to_num(y_pred)
	y_test = convert_onehots_to_num(y_test)
	#print(y_p)
	y_pred_v = getVoicedArray(y_p)
	y_test_v = getVoicedArray(y_test)
	y_pred_c = convertArrayToCents(y_p)
	y_test_c = convertArrayToCents(y_test)
	return mir_eval.melody.raw_chroma_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)


def rca(y_pred, y_test):
	y_pred = convert_onehots_to_num(y_pred)
	y_test = convert_onehots_to_num(y_test)
	y_pred_v = getVoicedArray(y_pred)
	y_test_v = getVoicedArray(y_test)
	y_pred_c = convertArrayToCents(y_pred)
	y_test_c = convertArrayToCents(y_test)
	return mir_eval.melody.raw_chroma_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)
def rca(y_pred, y_test):
	y_pred = convert_onehots_to_num(y_pred)
	y_test = convert_onehots_to_num(y_test)
	y_pred_v = getVoicedArray(y_pred)
	y_test_v = getVoicedArray(y_test)
	y_pred_c = convertArrayToCents(y_pred)
	y_test_c = convertArrayToCents(y_test)
	return mir_eval.melody.raw_chroma_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)
def oa(y_pred, y_test):
	y_pred = convert_onehots_to_num(y_pred)
	y_test = convert_onehots_to_num(y_test)
	y_pred_v = getVoicedArray(y_pred)
	y_test_v = getVoicedArray(y_test)
	y_pred_c = convertArrayToCents(y_pred)
	y_test_c = convertArrayToCents(y_test)
	return mir_eval.melody.overall_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)
def rpa(y_pred, y_test):
	y_pred = convert_onehots_to_num(y_pred)
	y_test = convert_onehots_to_num(y_test)
	y_pred_v = getVoicedArray(y_pred)
	y_test_v = getVoicedArray(y_test)
	y_pred_c = convertArrayToCents(y_pred)
	y_test_c = convertArrayToCents(y_test)
	return mir_eval.melody.raw_pitch_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)

def generate_random_one_hots():
	pitches = []
	for i in range(31):
		n = random.randrange(722)
		one_hot = []
		for j in range(722):
			if j == n:
				one_hot.append(1)
				#print(j)
			else:
				one_hot.append(0)
		#print(pitches)
		pitches.append(one_hot)
	return pitches
def isVoiced(arr):
	if arr[721] == 1:
		return 1
	else: 
		return 0
def VoiceConfusionMatrix(y_pred, y_test):
	y_pred_v = np.array([isVoiced(xi) for xi in y_pred])
	y_test_v = np.array([isVoiced(xi) for xi in y_test])
	return sklearn.metrics.confusion_matrix(y_test_v, y_pred_v, labels = [0, 1])
def PitchConfusionMatrix(y_pred, y_test):
	y_pred = convert_onehots_to_num(y_pred)
	y_test = convert_onehots_to_num(y_test)
	return sklearn.metrics.confusion_matrix(y_test, y_pred, labels = range(722))

