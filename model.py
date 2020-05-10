# The class GAN and its functions of this file are in train.py to create a GAN model

# Import necessary packages
import tensorflow as tf

# Utils is another folder in main project directory which contains helper functions.
# ops file is imported which contains certain DC-GAN functions
from Utils import ops


class GAN:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''
	def __init__(self, options):
		
		# line 91 in train.py calls model.GAN(model_options).
		# These model_options are the assigned to self.options in this class.
		self.options = options

		# Creating batch normalization layers (from line 41-44 for generator and 46-49 for discriminator):
		# "batch normalization reduces the internal covariance shift"
		# It makes the learning of layers in the network more independent of each other.
		
		# The objective of batch norm layer is to make input to the activation layer, unit Gaussian.
		# So that neuron does not get saturate in case of sigmoid and tanh. 
		# It helps in the following:
				# Fast convergence of network.
				# Allows you to be care free about weight initialization.
				# Works as regularization.
		
		# Batch norm layers for generator
		self.g_bn0 = ops.batch_norm(name='g_bn0')
		self.g_bn1 = ops.batch_norm(name='g_bn1')
		self.g_bn2 = ops.batch_norm(name='g_bn2')
		self.g_bn3 = ops.batch_norm(name='g_bn3')

		# Batch norm layer for descriminator
		self.d_bn1 = ops.batch_norm(name='d_bn1')
		self.d_bn2 = ops.batch_norm(name='d_bn2')
		self.d_bn3 = ops.batch_norm(name='d_bn3')
		self.d_bn4 = ops.batch_norm(name='d_bn4')


	def build_model(self):
		# Functions that puts together the model parameters to build the model.

		img_size = self.options['image_size']	# image_size = 64

		# tf.placeholder is used to feed in training examples.
		# It takes first first argument as input tensor type, second argument shape of tensor to be fed in (which is [64,64,64,3])
		# Third argument is the name indicating which input will the place holder be used for.

		# Place holder for feeding in real images
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
		# Place holder for feeding in fake images
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
		# Place holder for feeding in real caption
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		# Place holder for feeding in the noise to generator
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])

		# generating a fake image, sending arguments as the noise (z) and the caption
		fake_image = self.generator(t_z, t_real_caption)
		
		# Send a real image and real caption to discriminator
		disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)

		# Send a wrong (but real) image and real caption to discriminator
		disc_wrong_image, disc_wrong_image_logits   = self.discriminator(t_wrong_image, t_real_caption, reuse = True)

		# Send a fake image and real caption to discriminator
		disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)
		

		# Generator loss - binary cross entropy 
		# calculated on basis of the fake image and real caption sent to discriminator
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image)))

		# Discriminator loss - binary cross entropy 
		# calculated over 3 different case.
		# on basis of the real image and real caption sent to discriminator

		d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
		# on basis of the wrong image and real caption sent to discriminator

		d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits,labels=tf.zeros_like(disc_wrong_image)))

		# on basis of the fake image and real caption sent to discriminator
		d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))

		# Total generator loss
		d_loss = d_loss1 + d_loss2 + d_loss3

		# trainable variables of generator and discriminator
		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]


		# Tensor as input
		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}

		# Trainable variables
		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		# Loss variables
		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'disc_real_image_logits' : disc_real_image_logits,
			'disc_wrong_image_logits' : disc_wrong_image,
			'disc_fake_image_logits' : disc_fake_image_logits
		}
		
		return input_tensors, variables, loss, outputs, checks


	def build_generator(self):
		img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		fake_image = self.sampler(t_z, t_real_caption)
		
		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}
		
		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

	# Sample Images for a text embedding
	def sampler(self, t_z, t_text_embedding):
		tf.get_variable_scope().reuse_variables()
		
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding],axis=1)
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train = False))
		
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train = False))
		
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train = False))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train = False))
		
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	# THIS IS THE GENERATOR!
	# This function takes in the noise and text embeddings and creates an image. 
	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		# image size by default is 64 x 64
		s = self.options['image_size']

		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		# ops.linear() takes in the text_embedding and text dimension
		# Leaky relu takes in x and return max of (x, leak*x)
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )

		# Concatenates tensors along one dimension.
		z_concat = tf.concat([t_z, reduced_text_embedding],axis=1)
		
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')

		# First layer, activation relu		
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0))
		
		# Second layer, activation relu
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1))
		
		# Third layer, activation relu
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2))
		
		# Four layer, activation relu
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3))
		
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		
		# Output layer activation tanh
		return (tf.tanh(h4)/2. + 0.5)


	# THIS IS THE DICRIMINATOR!
	# This function takes in an image and text embeddings and returns the sigmoid output after passing it through
	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		# Dicriminator layers:
		h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		h1 = ops.lrelu( self.d_bn1(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))) #16
		h2 = ops.lrelu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))) #8
		h3 = ops.lrelu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))) #4
		
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = tf.concat([h3, tiled_embeddings],axis=3, name='h3_concat')
		h3_new = ops.lrelu( self.d_bn4(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4
