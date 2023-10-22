"""
Functions to load the WAY-EEG-GAL dataset from csv files.

This code is adapted from the following repositories: 
	https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/datasets/_base.py#L460
	https://github.com/bitsofbits/kaggle_grasp_and_lift_eeg_detection/blob/master/code/grasp.py
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter, butter
from sklearn.utils import Bunch


def butter_lowpass(highcut, fs, order):
	# from https://github.com/bitsofbits/kaggle_grasp_and_lift_eeg_detection/blob/master/code/grasp.py
	nyq = 0.5 * fs
	high = highcut / nyq
	b, a = butter(order, high, btype="lowpass")
	return b, a


def butter_bandpass(lowcut, highcut, fs, order):
	# from https://github.com/bitsofbits/kaggle_grasp_and_lift_eeg_detection/blob/master/code/grasp.py
	nyq = 0.5 * fs
	cutoff = [lowcut / nyq, highcut / nyq]
	b, a = butter(order, cutoff, btype="bandpass")
	return b, a


def butter_highpass(highcut, fs, order):
	# from https://github.com/bitsofbits/kaggle_grasp_and_lift_eeg_detection/blob/master/code/grasp.py
	nyq = 0.5 * fs
	high = highcut / nyq
	b, a = butter(order, high, btype="highpass")
	return b, a


def get_path(subject, series, kind, train):
	# adapted from https://github.com/bitsofbits/kaggle_grasp_and_lift_eeg_detection/blob/master/code/grasp.py
	prefix = "./src/sparseeg/dataset/train" if train else "./src/sparseeg/dataset/test"
	return "{0}/subj{1}_series{2}_{3}.csv".format(prefix, subject, series, kind)


def load_series_data(series_list, subject=1, train=True):
	# adapted from https://github.com/bitsofbits/kaggle_grasp_and_lift_eeg_detection/blob/master/code/grasp.py
	# Returns: data, target, feature_names, target_names 
	loaded_data = []
	loaded_target = []
	# loop over series because we want the data concatenated into one large ndarray
	for series in series_list:
		data_path = get_path(subject, series, 'data', train)
		label_path = get_path(subject, series, 'classlabel', train)
		print(f"Loading data from csv files: {data_path}; {label_path}")
		df_data = pd.read_csv(data_path, index_col=0)
		df_labels = pd.read_csv(label_path, index_col=0)

		# extract target in type ndarray
		# Note: Jax requires the targets to be numerical
		series_target = df_labels['ClassLabel'].values
		loaded_target.append(series_target)

		# extract data in type ndarray and apply filters
		data = df_data.values
		# TODO: do we want to add min/max freq to config file and check what best numbers to use?
		min_freq = 0.2
		max_freq = 50
		# These are fixed from the data - do not change!
		filter_n = 4  # Order of the filters to use
		sample_rate = 500 
		print("Band pass filtering, f_l =", min_freq, "f_h =", max_freq)
		b, a = butter_bandpass(min_freq, max_freq, sample_rate, filter_n)                
		filtered_data = lfilter(b, a, data, axis=0)
		loaded_data.append(filtered_data)
	
	data = np.concatenate(loaded_data, axis=0)
	target = np.concatenate(loaded_target, axis=0)
	# get feature names (i.e., data columns) and target names (i.e., class labels)
	feature_names = df_data.columns.tolist()
	target_names = ['class{:02d}'.format(i) for i in range(13)]
	return data, target, feature_names, target_names


def load_wayeeggal(subject=1, train=True, return_X_y=False):
	"""Load and return the WAY-EEG-GAL dataset (classification).
	This function will load all series data from one subject. 

	This code is adapted from:
		https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/datasets/_base.py#L460

	The WAY-EEG-GAL dataset is a multi-class classification dataset.
		Paper introducing dataset: https://www.nature.com/articles/sdata201447

	Samples for Subject 1 Series 1-6: data.shape = (1185498, 32)
	Samples for Subject 2 Series 1-6: data.shape = (1410684, 32)
	Samples for Subject 3 Series 1-6: data.shape = (1105800, 32)
	Samples for Subject 4 Series 1-6: data.shape = (1159109, 32)
	Samples for Subject 5 Series 1-6: data.shape = (1253962, 32)
	Samples for Subject 6 Series 1-6: data.shape = (1250189, 32)
	Samples for Subject 7 Series 1-6: data.shape = (1367619, 32)
	Samples for Subject 8 Series 1-6: data.shape = (1116358, 32)
	Samples for Subject 9 Series 1-6: data.shape = (1191790, 32)
	Samples for Subject 10 Series 1-6: data.shape = (1227363, 32)
	Samples for Subject 11 Series 1-6: data.shape = (1245565, 32)
	Samples for Subject 12 Series 1-6: data.shape = (1305410, 32)


	=================   ==============
	Classes                         12
	Samples per class        [59,71,48] UPDATE ME
	Samples total                  178  UPDATE ME
	Dimensionality                  32
	Features             real, pos/neg
	=================   ==============

	Parameters
	----------
	subject : int, default=1
		The subject id to be loaded. Must be an integer from 1-12
	train : bool, default=True
		If True, series 1-6 data from the train folder will be loaded.
		If False, series 7 & 8 data from the test folder will be loaded.
	return_X_y : bool, default=False
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.

	Returns
	-------
	data : :class:`~sklearn.utils.Bunch`
		Dictionary-like object, with the following attributes.

		data : {ndarray, dataframe} of shape (178, 13)
			The data matrix. If `as_frame=True`, `data` will be a pandas
			DataFrame.
		target: {ndarray, Series} of shape (178,)
			The classification target. If `as_frame=True`, `target` will be
			a pandas Series.
		feature_names: list
			The names of the dataset columns.
		target_names: list
			The names of target classes.

	(data, target) : tuple if ``return_X_y`` is True
		A tuple of two ndarrays by default. The first contains a 2D array of shape
		(178, 13) with each row representing one sample and each column representing
		the features. The second array of shape (178,) contains the target samples.
	"""

	if train: # training csv files are series 1 through 6
		series_list = list(range(1,7))
	else: # testing csv files are series 7 and 8
		series_list = [7,8]

	data, target, feature_names, target_names = load_series_data(series_list, subject)

	if return_X_y:
		return data, target

	return Bunch(
		data=data,
		target=target,
		target_names=target_names,
		feature_names=feature_names,
	)


# @validate_params(
#     {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
#     prefer_skip_nested_validation=True,
# )