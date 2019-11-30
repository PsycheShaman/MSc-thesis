"""
This script is intended to be run from the command line to generate 'datasets' (see below definition) from a directory containing
pythonDict.txt files in any number and any depth of sub-directories.

A dataset in this context constitutes a directory containing 3 things:

1. Any number of track numpy arrays.
2. The same as above number of info_set numpy arrays (See below).
3. A single info.yaml file to help read the dataset.

An info_set contains all the associated information about a track in the form of a 16 length numpy array formatted as follow

label, num_tracklets, tracklet1_present, tracklet2_present, tracklet3_present, tracklet4_present, tracklet5_present, tracklet6_present, nsigmae, nsigmap, PT, dEdX, P, eta, theta, phi

The tracks and info_sets of a dataset are spreak out over 1 or many files each with the naming convention 'i_tracks.npy'
and 'i_info_set.npy.'
"""

import argparse, sys, os, shutil
import settings
import numpy as np
import yaml

#Usage python3 extract/dataset_generator.py train_small 10000 10000 4 note that num_electrons=-1, num_pions=-1 will result in all being exported.

parser=argparse.ArgumentParser()

parser.add_argument("name", type=str,
                    help="Name of the dataset.")
parser.add_argument("num_electrons", type=int,
                    help="Max number of electrons to add to the dataset, use -1 for unlimited.")
parser.add_argument("num_pions", type=int,
                    help="Max number of pions to add to the dataset use -1 for unlimited.")
parser.add_argument("min_tracklets", type=int,
                    help="Minimum number of tracklets for a valid track. Use 0 for all tracks.")
parser.add_argument('--minp', help='Minimum momentum.', type=float, default=settings.default_minp)
parser.add_argument('--maxp', help='Maximum momentum.', type=float, default=settings.default_maxp)
parser.add_argument('--num_tracks_per_file', type=int, help='Partition dataset up into files of length num_tracks_per_file. Use -1 for one file.', default=settings.default_num_tracklets_per_file)

args=parser.parse_args()

output_folder = settings.datasets_home_directory + '%s/' % args.name

if os.path.exists(output_folder) and os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

files = []
for r, d, f in os.walk(settings.python_dicts_directory):
    for file in f:
        if 'pythonDict.txt' in file:
            files.append(os.path.join(r, file))

#Info set structure
#label, num_tracklets, tracklet1_present, tracklet2_present, tracklet3_present, tracklet4_present, tracklet5_present, trackletsettings.num_tracklets_in_track_present, nsigmae, nsigmap, PT, dEdX, P, eta, theta, phi

tracklet_data_set = np.zeros((args.num_tracks_per_file, settings.num_tracklets_in_track) + settings.tracklet_shape, dtype='float32')
info_set = np.zeros((args.num_tracks_per_file, settings.info_set_size), dtype='float32')
track_count = 0
e_count = 0
p_count = 0
save_count = 0
total_track_count = 0

for i, fil in enumerate(files):
    print('Found %d / %d electrons and %d / %d pions valid tracks.' % (e_count, args.num_electrons, p_count, args.num_pions))
    print('Reading %d / %d' % (i, len(files)), fil.strip())
    f = open(fil)
    r = f.read()
    try:
        exec('dic = ' + r + '}')

        for track in dic.values():
            export = True

            if abs(track['pdgCode']) == 11:
                label = 1
                if e_count == args.num_electrons:
                    export=False
            elif abs(track['pdgCode']) == 211:
                label = 0
                if p_count == args.num_pions:
                    export=False
            else:
                print('UNRECOGNIZED pdgCode %d' % track['pdgCode'])
            num_tracklets = np.sum(['layer' in key for key in track.keys()])
            present_map = ['layer %d' % i in track for i in range(settings.num_tracklets_in_track)]
            nsigmae = track['nSigmaElectron']
            nsigmap = track['nSigmaPion']
            PT = track['PT']
            dEdX = track['dEdX']
            P = track['P']
            eta = track['Eta']
            theta = track['Theta']
            phi = track['Phi']

            if args.minp is not None and args.minp > P:
                export = False

            if args.maxp is not None and args.maxp < P:
                export = False

            if num_tracklets < args.min_tracklets:
                export = False

            if export:
                for track_no in range(num_tracklets):
                    arr = np.asarray(track['layer %d' % track_no], dtype='float32')

                    assert arr.shape == settings.tracklet_shape

                    if np.sum(arr) < 1e-4:
                        export_tracklet = False

                    tracklet_data_set[track_count % args.num_tracks_per_file][track_no] = arr
                    info_set[track_count % args.num_tracks_per_file][0] = label
                    info_set[track_count % args.num_tracks_per_file][1] = num_tracklets
                    info_set[track_count % args.num_tracks_per_file][2:8] = present_map
                    info_set[track_count % args.num_tracks_per_file][8] = nsigmae
                    info_set[track_count % args.num_tracks_per_file][9] = nsigmap
                    info_set[track_count % args.num_tracks_per_file][10] = PT
                    info_set[track_count % args.num_tracks_per_file][11] = dEdX
                    info_set[track_count % args.num_tracks_per_file][12] = P
                    info_set[track_count % args.num_tracks_per_file][13] = eta
                    info_set[track_count % args.num_tracks_per_file][14] = theta
                    info_set[track_count % args.num_tracks_per_file][15] = phi

                if label == 1:
                    e_count += 1
                else:
                    p_count += 1

                track_count += 1
                total_track_count += 1

            if (export and track_count % args.num_tracks_per_file == 0) or (e_count==args.num_electrons and p_count==args.num_pions):
                print('Saving %s' % (output_folder + '%d_*.npy' % (save_count)))
                np.save(output_folder + "%d_tracks.npy" % (save_count), tracklet_data_set[:track_count])
                np.save(output_folder + "%d_info_set.npy" % (save_count), info_set[:track_count])

                track_count = 0
                save_count += 1

                tracklet_data_set[::] = 0

            if (e_count == args.num_electrons and p_count == args.num_pions):
                break

        if (e_count==args.num_electrons and p_count==args.num_pions):
            break

    except Exception as e:
        print(e)

info = {}
info['total_num_tracks'] = total_track_count
info['num_electrons'] = e_count
info['num_pions'] = p_count
info['num_save_files'] = save_count

print(info)

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

output = yaml.dump(info, open(output_folder + 'info.yaml', 'w'), Dumper=Dumper)