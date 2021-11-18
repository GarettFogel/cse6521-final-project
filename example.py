import medleydb as mdb

# Load all multitracks
mtrack_generator = mdb.load_all_multitracks()
#for mtrack in mtrack_generator:
    #print(mtrack.track_id)
    # do stuff

# Load specific multitracks
mtrack1 = mdb.MultiTrack('LizNelson_Rainfall')
mtrack2 = mdb.MultiTrack('Phoenix_ScotchMorris')

# Look at some attributes
print(mtrack1.has_bleed)
print(mtrack1.stem_instruments)
print(mtrack1.melody1_annotation[:5])

# Attributes of a stem
example_stem = mtrack1.stems[1]
print(example_stem.instrument)

#loading subset of data
track_ids = ['MusicDelta_Rock', 'MusicDelta_Reggae', 'MusicDelta_Disco']
dataset_subset = mdb.load_multitracks(track_ids)

#all files for a specific instrument
clarinet_files = mdb.get_files_for_instrument('clarinet')

# get all valid instrument labels
instruments = mdb.get_valid_instrument_labels()
print(instruments)
