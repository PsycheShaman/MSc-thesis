# Python ALICE TRD data tools

This package provides code to extract and handle named datasets with specifiable filters from pythonDict files.

## Getting Started

Start off by ensuring that the settings.py file contains the desired settings. The variable names should be clear on what goes where. Although you are welcome to change the relevant directories as specified in the settings.py file, it is recommended that you work with the provided default directories. These directories have already been added to the .gitignore and so you do not need to worry about using these directories.

### The raw_data directory

The raw_data directory should contain all pythonDict files you would like to process in any directory structure you like (it is searched recursively for pythonDicts).

### The datasets_home directory

This directory serves as the parent directory for any 'datasets' used in your code.

#### What is a dataset?

A dataset in this context constitutes a directory containing 3 types of things:

1. Any number of track numpy arrays (See below).
2. The same as above number of info_set numpy arrays (See below).
3. A single info.yaml file to help read the dataset (See below).

A track numpy array is a n x 6 x (settings.tracklet_shape) array of float32 numbers which represents n tracks in the way you would expect.

An info_set contains all the associated information about a track in the form of a 16 length numpy array arranged as follows:

label, num_tracklets, tracklet1_present, tracklet2_present, tracklet3_present, tracklet4_present, tracklet5_present, tracklet6_present, nsigmae, nsigmap, PT, dEdX, P, eta, theta, phi.

The tracks and info_sets of a dataset are spread out over 1 or many files each with the naming convention 'i_tracks.npy' and 'i_info_set.npy'. The reason for this is primarily for future memory considerations.

The info.yaml file is a human readable however still computer parse-able file which gives some overall info about the dataset.

## Using the extract.dataset_generator.py script

This script is intended to be run from the command line to generate 'datasets' (see above definition) from a directory containing pythonDict.txt files in any number and any depth of sub-directories.

Below is the output of `python extract/dataset_generator.py -h`

```
usage: dataset_generator.py [-h] [--minp MINP] [--maxp MAXP]
                            [--num_tracks_per_file NUM_TRACKS_PER_FILE]
                            name num_electrons num_pions min_tracklets

positional arguments:
  name                  Name of the dataset.
  num_electrons         Max number of electrons to add to the dataset, use -1
                        for unlimited.
  num_pions             Max number of pions to add to the dataset use -1 for
                        unlimited.
  min_tracklets         Minimum number of tracklets for a valid track. Use 0
                        for all tracks.

optional arguments:
  -h, --help            show this help message and exit
  --minp MINP           Minimum momentum.
  --maxp MAXP           Maximum momentum.
  --num_tracks_per_file NUM_TRACKS_PER_FILE
                        Partition dataset up into files of length
                        num_tracks_per_file. Use -1 for one file.
```

example: `python extract/dataset_generator.py test 100 100 4 --num_tracks_per_file=10`

## The datatools package

This package is intended to provide the basic functions for interacting with datasets. For now, all modules in this package will be dumped into the datatools namespace, so you can simply import them from there.

### load_and_save.py

This module contains functions to load existing datasets and save new datasets. The module is designed to run on the settings provided in the settings.py module but these can be overriden.

It is intended that functions will exist to allow models to be trained and tested on datasets which are too large to fit into memory, however for now, the loading and saving functions have been written to act on all the data at once.

## Using Datasets

Below is an example of how to produce 2 datasets (one for training and one for testing) of equal size with equal numbers of pions and electrons from a bunch of pythonDict's in the raw_data folder.

1. python extract/dataset_generator.py test_all 100 100 4
2. Set `default_dataset = 'test_all'` in settings.py
3. Split the data up in any way you wish. For example in tests/split_dataset.py `python3 -i tests/split_dataset.py`.

There should now be 3 new datasets in your default dataset directory.

## Some notes

Please be careful when saving datasets (including dataset_generator.py) it will overwrite any existing dataset of the same name.

It may be desireable to convert datasets into a custom datastructure in the future.