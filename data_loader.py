# This is the data processing file to use the image captions and images to make skip-thought embeddings
# To run this in colab use command ->   !python data_loader.py --data_set=="flowers"

# Import necessary packages
import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import h5py
import time

# Importing skipthoughts which is a freely available package 
# taken from the github link: https://github.com/ryankiros/skip-thoughts
# The below import is importing the file "skipthougts.py" from the main project dir.
import skipthoughts


def save_caption_vectors_flowers(data_dir):
	# data_dir argument by default contains the "Data" directory from the main project folder.


	# Check if skipthoughts have been imported successfully
	skipthoughts.test()	#test() is a function avaiable in skipthoughts.py

	# handle to the image directory. ie; ./Data/flowers/jpg
	img_dir = join(data_dir, 'flowers/jpg')
	# Load all image files in a list if 'jpg' is in the file name.
	image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]

	print("No of Images: ",len(image_files))	# Print number of image file. Oxford102 into contains 8189 images.

	# Declare a dictionary for every image file.
	# Where the image file name is the key and the list of captions (initailly empty list) is the value.
	image_captions = { img_file : [] for img_file in image_files }

	print("Image captions dict: ",image_captions)   # Initally the dict has all image files with empty dict values.

	# handle to the captions directory. ie; ./Data/flowers/text_c10
	caption_dir = join(data_dir, 'flowers/text_c10')
	class_dirs = []

	# as there are 102 different classes of flowers in Oxford102 dataset, run a for loop across all the class folders.
	# Class folder names are of the format "Class_00001","Class_00050", etc.
	# The txt files under each class folder corresponds to the captions for an image of that class.
	for i in range(1, 103):
		class_dir_name = 'class_%.5d'%(i)
		class_dirs.append( join(caption_dir, class_dir_name))

	# class dirs list now conatins the path to every class folder.
	# eg: ['Data/flowers/text_c10/class_00001', 'Data/flowers/text_c10/class_00002',.....,'Data/flowers/text_c10/class_00102']
	print("class_dirs after iterating through all class folders:",class_dirs)

	# Iteration count variable
	i = 0
	# Run the for loop for every class in class_dirs. Hence run in 102 times.
	# class_dir iterator represents the path to i-th class folder i going from 1 to 102
	for class_dir in class_dirs:

		# caption_files is a list of all the caption files (.txt) files in the class_dir
		caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
		print("Caption_files in class_dirs '",class_dir,"' ---> ",caption_files)

		# for loop to iterate through every caption file for the i-th class directory
		for cap_file in caption_files:
			with open(join(class_dir,cap_file)) as f:
				# Read caption using f.read()
				captions = f.read().split('\n')
			
			# cap_file is the name of .txt caption file.
			# The caption in every .txt file corresponds to the image with the same name and extension.jpg
			# Hence find the image file by taking the first 11 charaters of the caption file and concat ".jpg"
			img_file = cap_file[0:11] + ".jpg"

			# In the image_captions dictionary, assign the 5 text captions for the corresponding image.
			image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

		# Do the same for all image files in a class. 
		print("Iteration:",i)
		i+=1
	# Do the same for all 102 classes in the dataset
	print("End of Image-Caption linking!")

	print(len(image_captions))

	# Create a skipthought model using load_model() function.
	model = skipthoughts.load_model()

	# Dictionary for the encoded captions, initally empty.
	encoded_captions = {}

	# i -> element number ; img -> "image_file".jpg where each image_file is of format image_000xx.jpg
	for i, img in enumerate(image_captions):
		# Start time at the beginning of for loop
		st = time.time()

		# encode the caption using skipthoughts.enocde() function which takes arguments - model and the text-caption list per image.
		# the key of encoded_captions is the image name and the value is the encoded captions.
		encoded_captions[img] = skipthoughts.encode(model, image_captions[img])

		# Print time take for the encoding
		print(i, len(image_captions), img)
		print("Seconds", time.time() - st)
		
	# Create a hdf5 file named "flowers_embeddings.hdf5" to save the image-caption embeddings
	h = h5py.File(join(data_dir, 'flowers_embeddings.hdf5'))	#The Hierarchical Data Format version 5

	for key in encoded_captions:
		# key -> image name
		# encoded_captions[key] -> encoded caption for that key.
		# Create a dataset where each entry is the image name and corresponding image caption 
		h.create_dataset(key, data=encoded_captions[key])
	h.close()
			
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train', help='train/val')

	parser.add_argument('--data_dir', type=str, default='Data', help='Data directory')
	
	parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')

	parser.add_argument('--data_set', type=str, default='flowers', help='Data Set : Flowers')
	
	args = parser.parse_args()
	
	if args.data_set == 'flowers':
		save_caption_vectors_flowers(args.data_dir)

	# else use some other data_set (Future scope.)

# To call the function from main driver
def startEmbeddings():
	main()

def temp():
    print("In data_loader.py")


if __name__ == '__main__':
	main()
