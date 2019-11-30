import numpy as np
import settings
import yaml
import os, shutil

def get_dataset_info(tracks, info_set, tracks_per_file):
	info = {}
	info['total_num_tracks'] = tracks.shape[0]
	info['num_electrons'] = int(np.sum(info_set[:,0]))
	info['num_pions'] = int(info['total_num_tracks'] - info['num_electrons'])
	info['num_save_files'] = int(np.ceil(info['total_num_tracks'] / tracks_per_file)) if tracks_per_file > 0 else 1

	return info

def delete_and_create_dataset_folder(name, datasets_home_directory=settings.datasets_home_directory):
	output_folder = datasets_home_directory + name + '/'
	if os.path.exists(output_folder) and os.path.isdir(output_folder):
		shutil.rmtree(output_folder)
	os.mkdir(output_folder)

def load_whole_named_dataset(name, datasets_home_directory=settings.datasets_home_directory):
	dataset_path = datasets_home_directory + name + '/'

	info = yaml.load(open(dataset_path + 'info.yaml'), Loader=yaml.Loader)

	tracks = np.zeros((info['total_num_tracks'],) + settings.track_shape, dtype='float32')
	info_set = np.zeros((info['total_num_tracks'], settings.info_set_size), dtype='float32')
	total_loaded = 0

	for i in range(info['num_save_files']):
		tracks_part = np.load(dataset_path + '%d_tracks.npy' % i)
		info_set_part = np.load(dataset_path + '%d_info_set.npy' % i)
		tracks[total_loaded:total_loaded + tracks_part.shape[0]] = tracks_part
		info_set[total_loaded:total_loaded + info_set_part.shape[0]] = info_set_part
		total_loaded += tracks_part.shape[0]

	print(info)

	return tracks, info_set

def save_dataset(name, tracks, info_set, tracks_per_file, datasets_home_directory=settings.datasets_home_directory):
	output_folder = datasets_home_directory + name + '/'

	info = get_dataset_info(tracks, info_set, tracks_per_file)

	delete_and_create_dataset_folder(name, datasets_home_directory)

	for i in range(info['num_save_files']):
		tracks_part = tracks[i*tracks_per_file:(i+1)*tracks_per_file]
		info_set_part = info_set[i*tracks_per_file:(i+1)*tracks_per_file]

		np.save(output_folder + '%d_tracks.npy' % i, tracks_part)
		np.save(output_folder + '%d_info_set.npy' % i, info_set_part)

	yaml.dump(info, open(output_folder + 'info.yaml', 'w'), Dumper=yaml.Dumper)
	return info

def load_whole_default_dataset():
	return load_whole_named_dataset(settings.default_dataset)