# This is the training file to train the GAN model
# To run this in colab use command ->   !python train.py --data_set=="flowers" epochs==10

# Import necessary packages
import tensorflow as tf
import numpy as np
import argparse
import pickle
from os.path import join
import h5py
import scipy.misc
import random
import json
import os
import shutil
import imageio

# model is the python file which contains generated and discriminator architecture.
import model
# Utils is another folder in main project directory which contains helper functions.
# image_processing file is imported for image manipulations before training.
from Utils import image_processing


def main():
	
	# argunment parser variable.
	parser = argparse.ArgumentParser()

	# add the following as possible arguments that can be passed while running the file.
	# [
	# 	d_dim (Noise dimension)
	# 	t_dim (Text feature dimension)
	# 	batch_size (No of images used in training during iterations)
	# 	gf_dim (neurons in generators first layer)
	# 	df_dim (neurons in discriminators first layer)
	# 	data_dir (Path to data directory)
	# 	learning rate 
	# 	beta1 (momentum value for adam update)
	# 	epochs (Number of epochs) **10 epoch take around 6-7 hours**
	# 	save_every (number of iterations over which the model is saved)
	# 	resume_model (to resume the training of a model from file)
	# 	data_set (which data set to train on)
	# ]
	
	parser.add_argument('--z_dim', type=int, default=100, help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default=256, help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')

	parser.add_argument('--image_size', type=int, default=64, help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=64, help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=64, help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024, help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=2400, help='Caption Vector Length')

	parser.add_argument('--data_dir', type=str, default="Data", help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5, help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=600, help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=30, help='Save Model/Samples every x iterations over batches')

	parser.add_argument('--resume_model', type=str, default=None, help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--data_set', type=str, default="flowers", help='Which data set?')

	args = parser.parse_args()

	# Dict defining the model properties depending upon the command line arguments. 
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length
	}
	
	# Call the model.GAN function from the model file and pass the above dictionary to create a model based on those properties.
	gan = model.GAN(model_options)
	# "gan" is the handle to that model for rest of the code

	# Unpacking the values sent by build_model() function
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	

	# Based on loss recieved from gan.build_model() use adam optimizer to minimize the loss
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):	
		d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
	
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):	
		g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
	

	# Initialize all variables
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	
	# If resuming a trained model for further training
	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)
	
	loaded_data = load_training_data(args.data_dir, args.data_set)
	
	j=0	#To keep track of iterations
	# For "args.epochs" number of epochs---
	for i in range(args.epochs):
		batch_no = 0
		print("Batch size: ",args.batch_size)
		print("loaded_data['data_length']: ",loaded_data['data_length'])	#6000
	
		while batch_no*args.batch_size < loaded_data['data_length']:

			print("batch_no:",batch_no+1,"iteration_no:",j+1,"epoch:",i+1)

			# Create a training batch which is fed into the dicriminator in the current batch.
			real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size, 
				args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
			
			# DISCR UPDATE
			check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]

			# Feed in input from the training batch using feed_dict to the placeholders
			_, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			
			# Print the discriminator losses
			print("d1", d1)
			print("d2", d2)
			print("d3", d3)
			print("D", d_loss)
			
			# GEN UPDATE
			# Feed in input from the training batch using feed_dict to the placeholders
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})

			# GEN UPDATE TWICE, to make sure d_loss does not go to 0
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			
			# Print final loss of current batch
			print("LOSSES", d_loss, g_loss, batch_no, i, len(loaded_data['image_list'])/ args.batch_size,"\n")
			batch_no += 1
			j+=1

			# Regularly save the network
			if (batch_no % args.save_every) == 0:
				print("Saving Images, Model","\n\n")
				save_for_vis(args.data_dir, real_images, gen, image_files)
				save_path = saver.save(sess, "Data/Models/latest_model_{}_temp.ckpt".format(args.data_set))

		if i%5 == 0:
			save_path = saver.save(sess, "Data/Models/model_after_{}_epoch_{}.ckpt".format(args.data_set, i))

# Load training data from the image-text embeddings
def load_training_data(data_dir, data_set):
	if data_set == 'flowers':
		h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
		flower_captions = {}

		for ds in h.items():
			flower_captions[ds[0]] = np.array(ds[1])
		image_list = [key for key in flower_captions]
		image_list.sort()

		img_75 = int(len(image_list)*0.75)
		training_image_list = image_list[0:img_75]

		# Shuffle all images to get diversity in training examples
		random.shuffle(training_image_list)
		
		return {
			'image_list' : training_image_list,
			'captions' : flower_captions,
			'data_length' : len(training_image_list)
		}
	
	else:
		with open(join(data_dir, 'meta_train.pkl')) as f:
			meta_data = pickle.load(f)
		# No preloading for MS-COCO
		return meta_data


# Function to save the images produced by generator in an batch
def save_for_vis(data_dir, real_images, generated_images, image_files):
	
	shutil.rmtree( join(data_dir, 'samples') )
	os.makedirs( join(data_dir, 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		imageio.imwrite( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		imageio.imwrite(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


# To randomly generate a training batch from the input data set
def get_training_batch(batch_no, batch_size, image_size, z_dim, 
	caption_vector_length, split, data_dir, data_set, loaded_data = None):

	if data_set == 'flowers':
		real_images = np.zeros((batch_size, 64, 64, 3))
		wrong_images = np.zeros((batch_size, 64, 64, 3))
		captions = np.zeros((batch_size, caption_vector_length))

		cnt = 0
		image_files = []
		for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
			idx = i % len(loaded_data['image_list'])
			image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
			image_array = image_processing.load_image_array(image_file, image_size)
			real_images[cnt,:,:,:] = image_array
			
			# Improve this selection of wrong image
			wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
			wrong_image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][wrong_image_id])
			wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
			wrong_images[cnt, :,:,:] = wrong_image_array

			random_caption = random.randint(0,4)
			captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
			image_files.append( image_file )
			cnt += 1

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files

# To call the function from main driver
def startTraining():
	main()

def temp():
    print("In train.py")


if __name__ == '__main__':
	main()
	print("Training complete!")