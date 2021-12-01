import mir_eval
import sklearn#.metrics.confusion_matrix
import sklearn.metrics
import numpy as np
import random
import librosa as lr
import dataset as ds
NUM_CLASSES = 49

def convert_onehots_to_num(one_hots):
    pitches = []
    for i in one_hots:
        j = np.array(i)
        x = np.where(j > 0)
        pitches.append(x[0])
    #print(pitches)
    return pitches

#def convertVarToHz(i):
#   midi =  (i/(NUM_CLASSES-1))*(83-36) + 36
#   hz = 2**((midi-69)/12)*440
#   if i == NUM_CLASSES-1:
#       return 0;
#   else:
#       return hz

def convertVarToHz(i):
    if i == NUM_CLASSES-1:
        return 0;
    else:
        note = ds.NOTES[i]
        return lr.note_to_hz(note)

def convertArrayToCents(arr):
    hzVals = np.array([convertVarToHz(xi) for xi in arr]).astype(float)
    #print(arr[2])
    hzVals = hzVals.reshape(-1)
    #print(hzVals)
    return mir_eval.melody.hz2cents(hzVals)

def if722(i):
    if i == NUM_CLASSES-1:
        return 0
    else:
        return 1

def getVoicedArray(arr):
    hzVals = np.array([if722(xi) for xi in arr])
    hzVals = hzVals.reshape(-1)
    return hzVals


def rca(y_pred, y_test):
    #print(y_p)
    y_pred_v = getVoicedArray(y_pred)
    y_test_v = getVoicedArray(y_test)
    y_pred_c = convertArrayToCents(y_pred)
    y_test_c = convertArrayToCents(y_test)
    return mir_eval.melody.raw_chroma_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)

def oa(y_pred, y_test):

    y_pred_v = getVoicedArray(y_pred)
    y_test_v = getVoicedArray(y_test)
    y_pred_c = convertArrayToCents(y_pred)
    y_test_c = convertArrayToCents(y_test)
    return mir_eval.melody.overall_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)

def rpa(y_pred, y_test):

    y_pred_v = getVoicedArray(y_pred)
    y_test_v = getVoicedArray(y_test)
    y_pred_c = convertArrayToCents(y_pred)
    y_test_c = convertArrayToCents(y_test)
    return mir_eval.melody.raw_pitch_accuracy(y_test_v, y_test_c, y_pred_v, y_pred_c)

def isVoiced(pred):
    if pred == NUM_CLASSES-1:
        return 0
    else:
        return 1

def VoiceConfusionMatrix(y_pred, y_test):
    y_pred_v = np.array([isVoiced(xi) for xi in y_pred])
    y_test_v = np.array([isVoiced(xi) for xi in y_test])
    return sklearn.metrics.confusion_matrix(y_test_v, y_pred_v, labels = [0, 1], normalize='true')

def PitchConfusionMatrix(y_pred, y_test):
    return sklearn.metrics.confusion_matrix(y_test, y_pred, labels = range(NUM_CLASSES),normalize='true')

'''

x = generate_random_one_hots()
y = x;
x = convert_onehots_to_num(x)
y = convert_onehots_to_num(y)
print(rca(x,y))
print(rpa(x,y))
print(oa(x,y))
print(VoiceConfusionMatrix(x, y))
print(PitchConfusionMatrix(x, y))
x = generate_random_one_hots()
y = generate_random_one_hots();
x = convert_onehots_to_num(x)
y = convert_onehots_to_num(y)
print(rca(x,y))
print(rpa(x,y))
print(oa(x,y))
print(VoiceConfusionMatrix(x, y))
print(PitchConfusionMatrix(x, y))'''
