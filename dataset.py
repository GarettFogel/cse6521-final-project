import medleydb as mdb
import librosa as lr
import numpy as np
import pandas as pd
from scipy import stats

# mtrack_generator = mdb.load_all_multitracks()
octave = lr.key_to_notes('C:maj')
NOTES = []
for oct in range(2,6): #do octaves 2-5
	NOTES += [x+str(oct) for x in octave]
NOTES = np.array(NOTES)

NO_VOICE = "None"

def get_train_test_songs():
	train_songs = np.squeeze(pd.read_csv(train_list.csv).values)
	test_songs = np.squeeze(pd.read_csv(test_list.csv))
	return train_songs, test_songs

def datify_track(track_name, preprocess=False):
	'''Tacks a track name and retruns x_data,ydata where 
	x_data is a an array of data windows 
	y_data is a an array of one hot vectors containin the label for each window as a one-hot vector'''
	#Handle to mdb processor
	mtrack = mdb.Multitrack(track_name)

	## Read/Format x data
	hop_size = 80 #hop size used by paper 
	window_size = 1024 #window size used by paper
	wav, samp_rate = lr.load(mtrack.mix_path, sr=8000) #8000 seems to be rate used by paper
	wav_length = wav.shape[0]
	# num_windows = int(wav_length/window_size)

	# remove last bit of audio to fit window size	
	# tail = wav_length % window_size
	# wav = wav[:-tail]

	# For regular nn, precprocess data using fourier transform and hanning window
	#create 2D array of audio chunks	
	if(preprocess):
		x_data = lr.stft(wav, n_fft=1024, hop_length=window_size, win_length=window_size, window='hann')
		x_data = np.transpose(x_data)
		x_data = np.abs(x_data) #magnitude of complex values
	else:
		x_data = lr.util.frame(wav, frame_length=window_size, hop_length=hop_size, axis=0)
		hann = lr.filters.get_window('hann', window_size, fftbins=False)
		x_data = x_data*hann #apply hann filter to each window
		# x_data = np.reshape(num_windows,window_size)

	## Read/Format y data
	# get id of dominant vocal stem
	vocal_stems = [stem_id for stem_id in range(len(mtrack.stems)) if is_singer(mtrack.stem[stem_id].instrument)]
	vocal_ranks = []
	melody_stems = list(mtrack.melody_rankings.keys())
	melodic_vocal_stems = [stem_id for stem_id in vocal_stems if stem_id in melody_stems]
	dominant_vocal_stem_id = 999999

	for stem_id in melodic_vocal_stems:
		rank = mtrack.melody_rankings[stem_id]
		if(rank < dominant_vocal_stem_id):
			dominant_vocal_stem_id = stem_id

	#get labels for that vocal stem
	if(dominant_vocal_stem_id = 999999):
		print("No dominant vocal track for " track_name)
		return -1

	annos = np.array(mtrack.stems[dominant_vocal_stem_id].pitch_annotation)
	anno_times = annos[:,0]
	anno_freqs = annos[:,1]

	# get median pitch label for each window
	num_windows = x_data.shape[0]
	y_data = np.empty((num_windows,num_classes))
	for i in range(num_windows):
		start_time = i*hop_size / samp_rate
		end_time = (i*window_size+window_size) / samp_rate

		#get frequency labels which occur during window time
		win_freqs = anno_freqs[np.where(np.logical_and(anno_times >= start_time, anno_times <= end_time))[0]]
		win_notes=get_pitch_labels(win_freqs)
		dominant_note = np.mode(win_notes) #May want to change this
		y_data[i] = get_note_one_hot(dominant_note)


	return x_data, y_data

# Check if instrument is a singer
def is_singer(instrument):
	return instrument == "female singer" or instrument == "male singer"

#Convert frequencies to note names (slightly different than paper)
def get_pitch_label(freqs):
	notes = lr.hz_to_note(freq)	
	return np.array([x for x in notes if x in NOTES else NO_VOICE])

def get_note_one_hot(note):
	num_classes = NOTES.shape[0]
	note_pos = np.where(NOTES==note)
	return one_hot(note_pos,num_classes)

def one_hot(pos,num_classes)
	vec = np.zeros(num_classes)
	vec[pos] = 1
	return vec