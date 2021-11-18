import medleydb as mdb
import librosa as lr
import numpy as np
import pandas as pd

# mtrack_generator = mdb.load_all_multitracks()

def get_train_test_songs():
	train_songs = np.squeeze(pd.read_csv(train_list.csv).values)
	test_songs = np.squeeze(pd.read_csv(test_list.csv))
	return train_songs, test_songs

def is_singer(instrument):
	return instrument == "female singer" or instrument == "male singer"

def datify_track(track_name):
	#Handle to mdb processor
	mtrack = mdb.Multitrack(track_name)

	## Read/Format x data
	window_size = 80 #window size used by paper
	wav, samp_rate = lr.load(mtrack.mix_path, sr=8000) #8000 seems to be rate used by paper
	wav_length = wav.shape[0]
	num_windows = int(wav_length/window_size)

	# remove last bit of audio to fit window size	
	tail = wav_length % window_size
	wav = wav[:-tail]

	#create 2D array of audio chunks	
	x_data = np.reshape(num_windows,window_size)

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
	y_data = np.empty(num_windows)
	for i in range(num_windows):
		start_time = i*window_size / samp_rate
		end_time = (i+1)*window_size / samp_rate

		#get frequency labels which occur during window time
		labels = anno_freqs[np.where(np.logical_and(anno_times >= start_time, anno_times <= end_time))[0]]
		label = np.median(labels) #May want to change this
		y_data[i] = label


	return x_data, y_data
