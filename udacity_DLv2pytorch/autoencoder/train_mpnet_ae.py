import argparse
import sys, os, time
import fnmatch

import numpy as np 

import torch.nn as nn 
import torch.nn.functional as F 

from tools.import_tool import fileImport

def load_normalized_dataset(env_names,pcd_data_path,importer,min_length=(5351*3)):
	"""
	Load point cloud dataset into array of obstacle pointclouds, which will be entered as input to the encoder NN, but first normalizing all the data based on mean and norm

	Input: 	env_names (list) - list of strings with names of environments to import
			pcd_data_path (string) - filepath to file with environment representation
			importer (fileImport) - object from utility library to help with importing different data
			min_length (int) - if known in advance, number of flattened points in the shortest obstacle point cloud vector

	Return: obstacles (numpy array) - array of obstacle point clouds, with different rows for different environments
										and different columns for points
	"""
	# get file names, just grabbing first one available (sometimes there's multiple)
	fnames = []

	print("Searing for file names...")
	for i, env in enumerate(env_names):
		# hacky reordering so that we don't load the last .pcd file which is always corrupt
		# sort by the time step on the back, which helps us obtain the earliest possible
		for file in sorted(os.listdir(pcd_data_path), key=lambda x: int(x.split('Env_')[1].split('_')[1][:-4])):
			if (fnmatch.fnmatch(file, env+"*")):
				fnames.append(file)
				break

	if min_length is None: # compute minimum length for dataset will take twice as long if necessary
		min_length = 1e6 # large to start
		for i, fname in enumerate(fnames):
			length = importer.pointcloud_length_check(pcd_fname=pcd_data_path + fname)
			if (length < min_length):
				min_length = length

	print("Loading files, minimum point cloud obstacle length: ")
	print(min_length)
	N = len(fnames)

	# make empty array of known length, and use import tool to fill with obstacle pointcloud data
	min_length_array = int(min_length/3)
	obstacles_array = np.zeros((3, min_length_array, N), dtype=np.float32)
	for i, fname in enumerate(fnames):
		data = importer.pointcloud_import_array(pcd_data_path + fname, min_length_array) #using array version of import, and will flatten manually after normalization
		obstacles_array[:, :, i] = data

	# compute mean and std of each environment
	means = np.mean(obstacles_array, axis=1)
	stds = np.std(obstacles_array, axis=1)
	norms = np.linalg.norm(obstacles_array, axis=1)

	# compute mean and std of means and stds
	mean_overall = np.expand_dims(np.mean(means, axis=1), axis=1)
	std_overall = np.expand_dims(np.std(stds, axis=1), axis=1)
	norm_overall = np.expand_dims(np.mean(norms, axis=1), axis=1)

	print("mean: ")
	print(mean_overall)
	print("std: ")
	print(std_overall)
	print("norm: ")
	print(norm_overall)

	# normalize data based on mean and overall norm, and then flatten into vector
	obstacles=np.zeros((N,min_length),dtype=np.float32)
	for i in range(obstacles_array.shape[2]):
		temp_arr = (obstacles_array[:, :, i] - mean_overall)
		temp_arr = np.divide(temp_arr, norm_overall)
		obstacles[i] = temp_arr.flatten('F')

	return obstacles

class MPNetAutoEncoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(MPNetEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 256), nn.PReLU(),
        nn.Linear(256,256), nn.PReLU(),
        nn.Linear(256, encoding_dim))

        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 256), nn.PReLU(),
        nn.Linear(256,256), nn.PReLU(),
        nn.Linear(256, input_size))
    
    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y


def main(args):

    env_data_path = args.env_data_path    
    importer = fileImport()

    env_names = importer.environments_import(env_data_path + args.envs_file)
    pcd_path = args.pcd_data_path
    obstacles = load_normalized_dataset(env_names, pcd_path, importer)

    ## mpnet AE
    pcd_inp_size = args.enc_input_size
    latent_size = args.latent_size
    mpnae = MPNetAutoEncoder(pcd_inp_size, latent_size)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--pcd_data_path', type=str, default="D:\\SL\\projects\\2020\\May\\mpnet_ws\\src\\baxter_mpnet_experiments\\data\\train\\pcd\\")
    parser.add_argument('--env_data_path', type=str, default="D:\\SL\\projects\\2020\\May\\mpnet_ws\\src\\baxter_mpnet_experiments\\env\\environment_data\\")
    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--latent_size', type=int, default=60)

    args = parser.parse_args()
    main(args)