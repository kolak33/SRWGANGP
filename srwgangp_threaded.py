import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Add, add, Layer, AveragePooling2D
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, Convolution2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
from keras.applications.vgg19 import VGG19
import csv
from config_srwgangp_threaded import SRWGANGPConfig
from keras.models import load_model
import random
from multiprocessing import JoinableQueue
from multiprocessing import Process
import cv2
from scipy import misc
from skimage.measure import compare_ssim as ssim

#
#
#	parts of code come from:
#	https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
#	and
#	https://github.com/JGuillaumin/SuperResGAN-keras/blob/master/SRGAN-VGG54.ipynb
#
#

# used in gradient penalty loss
class RandomWeightedAverage(_Merge, SRWGANGPConfig):
	def _merge_function(self, inputs):
		print('Weighted avg.')
		weights = K.random_uniform((self.BATCH_SIZE, 1, 1, 1))
		return (weights * inputs[0]) + ((1 - weights) * inputs[1])
		
class SRWGANGP(SRWGANGPConfig):
	def __init__(self):
		self.num_training_images = len(os.listdir(self.LR_TRAIN_IMAGES_PATH))
		self.num_validation_images = len(os.listdir(self.LR_VALID_IMAGES_PATH))
		self.num_test_images = len(os.listdir(self.LR_TEST_IMAGES_PATH))

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def gradient_penalty_loss(self, y_true, y_pred, averaged_samples,
							  gradient_penalty_weight):
		# first get the gradients:
		#   assuming: - that y_pred has dimensions (batch_size, 1)
		#             - averaged_samples has dimensions (batch_size, nbr_features)
		# gradients afterwards has dimension (batch_size, nbr_features), basically
		# a list of nbr_features-dimensional gradient vectors

		gradients = K.gradients(y_pred, averaged_samples)[0]
		# compute the euclidean norm by squaring ...
		gradients_sqr = K.square(gradients)
		#   ... summing over the rows ...
		gradients_sqr_sum = K.sum(gradients_sqr,
					axis=np.arange(1, len(gradients_sqr.shape)))
		#   ... and sqrt
		gradient_l2_norm = K.sqrt(gradients_sqr_sum)
		# compute lambda * (1 - ||grad||)^2 still for each single sample
		gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
		# return the mean as loss over all the batch samples
		return K.mean(gradient_penalty)

	# PSNR metric
	def PSNR(self, y_true, y_pred):
		return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

	# build a residual block
	def res_block(self, inputs):
		x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(inputs)
		x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
		x = add([x, inputs])
		return x

	# build an upscale block
	def up_block(self, x):
		x = UpSampling2D(size=(2, 2))(x)
		x = Conv2D(256, kernel_size=(3,3), strides=(1,1) , padding='same', activation='relu')(x)
		return x
		
	def make_generator(self):
		NUM_RESIDUAL_BLOCKS = 16
		
		# Low resolution image input
		input_generator = Input(shape=(None, None, 3), name='input_generator')

		# Pre-residual block
		c1 = Conv2D(filters=64, kernel_size=(9,9), strides=(1,1), padding='same', activation=None)(input_generator)

		# Propogate through residual blocks
		r = self.res_block(c1) 
		for _ in range(NUM_RESIDUAL_BLOCKS - 1):
			r = self.res_block(r)

		# Post-residual block
		c2 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(r)

		# Skip connection
		x = add([c2, c1])

		# Two upscale blocks
		u1 = self.up_block(x)
		u2 = self.up_block(u1)

		# final conv layer : activated with tanh -> pixels in [-1, 1]
		output_generator = Conv2D(3, kernel_size=(9,9), strides=(1,1), padding='same', activation='tanh')(u2)

		generator = Model(inputs=input_generator, outputs=output_generator)
		return generator
		
	def discriminator_block(self, layer_input, filters, strides):
		d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
		d = LeakyReLU(alpha=0.2)(d)
		return d

	# discriminator based on SRResNet model with removed Batch Normalisation layers
	def make_discriminator(self):
	# number of filters is lowered a bit, to reduce the size of the network
	# normally num_filters = 64
		num_filters = 32
		d0 = Input(shape=self.HR_SHAPE, name='input_discriminator')
		
		# number of filters in layers is also changed a bit
		d1 = self.discriminator_block(d0, num_filters*2, strides=1)
		d2 = self.discriminator_block(d1, num_filters*2, strides=2)
		
		d3 = self.discriminator_block(d2, num_filters*4, strides=1)
		d4 = self.discriminator_block(d3, num_filters*4, strides=2)
		
		d5 = self.discriminator_block(d4, num_filters*4, strides=1)
		d6 = self.discriminator_block(d5, num_filters*4, strides=2)
		
		d7 = self.discriminator_block(d6, num_filters*8, strides=1)
		d8 = self.discriminator_block(d7, num_filters*8, strides=2)

		flatten = Flatten(data_format='channels_last')(d8)
		
		d9 = Dense(num_filters*8)(flatten)
		d10 = LeakyReLU(alpha=0.2)(d9)
		
		# output without activation
		output = Dense(1, activation=None)(d10)

		discriminator = Model(d0, output)
		return discriminator

	# returns generator and discriminator models
	# creates validation and pretrained models
	def create_model_architecture(self, HR_TARGET_SIZE):
		HR_SHAPE = HR_TARGET_SIZE + (3,)
		print('HR shape:', HR_SHAPE)

		generator = 1
		discriminator = 1
		# initialize the generator and discriminator.
		generator = self.make_generator()
		discriminator = self.make_discriminator()

		# discriminator in generator_model is used for calculating wasserstein_loss
		# and it's layers are frozen
		for layer in discriminator.layers:
			layer.trainable = False
		discriminator.trainable = False

		generator_input = Input(shape=self.LR_SHAPE)
		generator_layers = generator(generator_input)
		discriminator_layers_for_generator = discriminator(generator_layers)

		generator_model_train = Model(inputs=[generator_input],
								outputs=[discriminator_layers_for_generator, generator_layers, generator_layers])
		
		loss=[self.wasserstein_loss, self.vgg_loss, 'mean_squared_error']
		loss_weights=[1e-3, 3e-1, 7e-1]

		generator_model_train.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
								loss=loss,
								loss_weights=loss_weights,
								metrics=[self.PSNR])
		generator_model_train.summary()

		# create generator model and pretrain it for PRETRAIN_EPOCHS
		if self.PRETRAIN_GENERATOR:
			self.pretrained_generator_model = Model(inputs=[generator_input],
									outputs=[generator_layers])
			self.pretrained_generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
									loss=['mean_squared_error'],
									metrics=[self.PSNR])

		self.generator_model_valid = Model(inputs=[generator_input],
								outputs=[discriminator_layers_for_generator, generator_layers, generator_layers])
		self.generator_model_valid.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
								loss=[self.wasserstein_loss, self.vgg_loss, 'mean_squared_error'],
								loss_weights=[1e-3, 3e-1, 7e-1],
								metrics=[self.PSNR])

		# generator_model is compiled, it's layers are now frozen and used in discriminator loss
		for layer in discriminator.layers:
			layer.trainable = True
		for layer in generator.layers:
			layer.trainable = False
		discriminator.trainable = True
		generator.trainable = False

		real_samples = Input(shape=HR_SHAPE)
		generator_input_for_discriminator = Input(shape=self.LR_SHAPE)
		generated_samples_for_discriminator = generator(generator_input_for_discriminator)
		discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
		discriminator_output_from_real_samples = discriminator(real_samples)

		# compute weighted-average of real and generated samples,
		# to use for the gradient norm penalty.

		averaged_samples = RandomWeightedAverage()([real_samples,
							generated_samples_for_discriminator])
		averaged_samples_out = discriminator(averaged_samples)

		partial_gp_loss = partial(self.gradient_penalty_loss,
					averaged_samples=averaged_samples,
					gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT)
		# functions need names or Keras will throw an error
		partial_gp_loss.__name__ = 'gradient_penalty'

		discriminator_model = Model(inputs=[real_samples,
							generator_input_for_discriminator],
						outputs=[discriminator_output_from_real_samples,
							 discriminator_output_from_generator,
							 averaged_samples_out])

		discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
								loss= [self.wasserstein_loss,
								       self.wasserstein_loss,
								       partial_gp_loss])
		discriminator_model.summary()
		
		return generator_model_train, discriminator_model

	def train_epoch_thread_pretrain(self, queue_batches, epoch):
		generator_loss_epoch_sum = 0.0
		generator_psnr_epoch_sum = 0.0	
		for current_iter in range(self.num_batches):			
			input_images, hd_images = queue_batches.get()

			generator_loss = self.pretrained_generator_model.train_on_batch(input_images, hd_images)
			generator_loss_epoch_sum = generator_loss_epoch_sum + generator_loss[0]
			generator_psnr_epoch_sum = generator_psnr_epoch_sum + generator_loss[1]
			
			self.curr_iter_train += 1
			if(self.curr_iter_train % 1000 == 0):
				print('PSNR train:', generator_loss[1])

			if ((current_iter+1) == self.num_batches):
				generator_psnr_epoch = generator_psnr_epoch_sum / self.num_batches
				generator_loss_epoch = generator_loss_epoch_sum / self.num_batches

				print('PSNR train epoch: ', generator_psnr_epoch)
			queue_batches.task_done()

	def train_epoch_thread_final_training(self, queue_batches, epoch):
		# opening csv files for saving results
		self.csv_train_final = open('results_srwgangp_train.csv', 'a')
		self.csv_train_epoch_final = open('results_srwgangp_train_epoch.csv', 'a')

		# set fake (generator's) labels as 1's
		# and true labels as -1's
		fake_y = np.ones((self.BATCH_SIZE, 1), dtype=np.float32)
		real_y = -fake_y
		dummy_y = np.zeros((self.BATCH_SIZE, 1), dtype=np.float32)

		num_batches_per_generator_in_epoch = self.num_batches // (self.TRAINING_RATIO + 1)
		num_batches_per_discr_in_epoch = num_batches_per_generator_in_epoch * self.TRAINING_RATIO

		current_iter = 0
		processed_batches = 0
		discriminator_loss_epoch_sum = 0.0
		generator_loss_epoch_sum = 0.0
		generator_psnr_epoch_sum = 0.0
		disc_loss = 5.
		for _ in range(int(num_batches_per_generator_in_epoch)):
			for j in range(self.TRAINING_RATIO):
				input_images, hd_images = queue_batches.get()
				current_batch_size = hd_images.shape[0]
				
				discr_loss_train_ratio = 0.0
				discr_loss = self.discriminator_model.train_on_batch(
					[hd_images, input_images],
					[real_y, fake_y, dummy_y])
				discr_loss_train_ratio = discr_loss_train_ratio + discr_loss[0]
				disc_loss = discr_loss
				processed_batches += 1
				queue_batches.task_done()

			discr_loss_avg = discr_loss_train_ratio / self.TRAINING_RATIO

			input_images, hd_images = queue_batches.get()
			current_batch_size = hd_images.shape[0]

			inputs_loss = [real_y, hd_images, hd_images]

			generator_loss = self.generator_model_train.train_on_batch(input_images, inputs_loss)
			generator_loss_epoch_sum = generator_loss_epoch_sum + generator_loss[0]

			generator_psnr_epoch_sum = generator_psnr_epoch_sum + generator_loss[5]
			discriminator_loss_epoch_sum = discriminator_loss_epoch_sum + discr_loss_avg
					
			current_iter += 1
			self.curr_iter_train += 1
			if (self.curr_iter_train % 200 == 0):
				csv_train_row = str(self.curr_iter_train) + ' ' + str(discr_loss_avg) + ' ' + str(generator_loss[0]) + ' ' + str(generator_loss[5]) + '\n'
				self.csv_train_final.write(csv_train_row)

				if(self.curr_iter_train % 1000 == 0):
					print('gen psnr:', generator_loss[5])
					print('disc loss:', disc_loss) 			
			processed_batches += 1

			if processed_batches == self.num_batches:
				# calculate epoch losses
				generator_loss_epoch = generator_loss_epoch_sum / num_batches_per_generator_in_epoch
				generator_psnr_epoch = generator_psnr_epoch_sum / num_batches_per_generator_in_epoch
				discriminator_loss_epoch = discriminator_loss_epoch_sum / num_batches_per_discr_in_epoch

				print('DISC train loss:', discriminator_loss_epoch)
				print('GEN train loss:', generator_loss_epoch)
				print('PSNR train: ', generator_psnr_epoch)

				self.no_improvement_wasserstein_loss += 1
				if self.best_discriminator_loss < discriminator_loss_epoch:
					self.save_weights(epoch)
					self.best_discriminator_loss = discriminator_loss_epoch
					self.no_improvement_wasserstein_loss = 0
					print('saved weights, discriminator loss improved.')
				# saving metrics and losses to csv file
				# format: epoch, discriminator_train_loss, generator_train_loss, generator_train_PSNR\n with ' ' delimiter
				csv_epoch_row = str(epoch) + ' ' + str(discriminator_loss_epoch) + ' ' + str(generator_loss_epoch) + ' ' + str(generator_psnr_epoch) + '\n'
				self.csv_train_epoch_final.write(csv_epoch_row)
				self.csv_train_final.close()
				self.csv_train_epoch_final.close()
				# decrease learning rate
				if (self.no_improvement_wasserstein_loss == self.LR_DECAY_NO_IMPROVEMENT_EPOCHS):
					lr_gen = K.get_value(self.generator_model_train.optimizer.lr)
					lr_disc = K.get_value(self.discriminator_model.optimizer.lr)

					print('Learning rate decreased from: ', lr_gen)
					lr_gen = lr_gen / self.LR_DECAY_FACTOR
					lr_disc = lr_disc / self.LR_DECAY_FACTOR

					K.set_value(self.generator_model_train.optimizer.lr, lr_gen)
					K.set_value(self.discriminator_model.optimizer.lr, lr_disc)
					print('Learning rate decreased to: ', lr_gen)
					self.no_improvement_wasserstein_loss = 0
			queue_batches.task_done()

	def train_epoch_thread(self, queue_batches, epoch, final_training):
		if not final_training:
			self.train_epoch_thread_pretrain(queue_batches=queue_batches, epoch=epoch)
		else:
			self.train_epoch_thread_final_training(queue_batches=queue_batches, epoch=epoch)

	def valid_epoch_thread(self, queue_batches, epoch, final_training, test_mode=False):
		print('Start validation')
		self.csv_valid_epoch = open('results_srwgangp_valid_epoch.csv', 'a')

		psnr_sum = 0.0
		#mssim_sum = 0.0
		for processed_batches in range(self.num_batches):
			input_images, hd_images = queue_batches.get()
			predictions = self.generator_model_valid.predict_on_batch(input_images)

			[_, hr_generated, _] = predictions

			psnr_val_batch = self.PSNR_val(y_true=hd_images, y_pred=hr_generated)
			psnr_sum = psnr_sum + psnr_val_batch
			#mssim_sum += self.mssim(y_true=hd_images, y_pred=hr_generated)
			if ((processed_batches+1) == self.num_batches):
				psnr_validation = psnr_sum / self.num_batches
				#mssim_valid = mssim_sum / self.num_batches
				#print('MSSIM validation: ', mssim_valid)

				if not test_mode:
					valid_title_row = str(epoch) + ' ' + str(psnr_validation) + '\n'
					self.csv_valid_epoch.write(valid_title_row)
					print('PSNR validation: ', psnr_validation)
				else:
					print('PSNR test: ', psnr_validation)

				if not test_mode and self.PRETRAIN_GENERATOR and not final_training and (psnr_validation > self.best_psnr_pretrain):
					print('Saving new pretrained generator')
					self.best_psnr_pretrain = psnr_validation
					self.pretrained_generator_model.save(self.PRETRAINED_GENERATOR_MODEL_PATH)
					self.pretrained_generator_model.save_weights(self.PRETRAINED_GENERATOR_WEIGHTS_PATH)
					self.no_improvement_pretrain_epochs = 0
								
				elif not test_mode and self.PRETRAIN_GENERATOR and not final_training:
					self.no_improvement_pretrain_epochs += 1

					if(self.no_improvement_pretrain_epochs == self.LR_DECAY_NO_IMPROVEMENT_EPOCHS):
						lr_pretrain = K.get_value(self.pretrained_generator_model.optimizer.lr)

						print('Learning rate decreased from: ', lr_pretrain)
						lr_pretrain = lr_pretrain / self.LR_DECAY_PRETRAIN

						K.set_value(self.pretrained_generator_model.optimizer.lr, lr_pretrain)
						print('Learning rate decreased to: ', lr_pretrain)
						self.no_improvement_pretrain_epochs = 0
				self.csv_valid_epoch.close()
			queue_batches.task_done()

	def data_gen_thread(self, queue, thread_id, file_path_list_lr, file_path_list_hr, train_step, final_training):
		start_index = self.num_batches_per_thread * self.BATCH_SIZE * thread_id
		for i in range(self.num_batches_per_thread):
			lr_batch = []
			hr_batch = []
			for j in range(self.BATCH_SIZE):
				index = start_index + (i * self.BATCH_SIZE) + j

				lr_img = cv2.imread(file_path_list_lr[index], cv2.COLOR_BGR2RGB)
				hr_img = cv2.imread(file_path_list_hr[index], cv2.COLOR_BGR2RGB)
				# random flip while training
				if train_step and np.random.random() < 0.5:
					lr_img = np.fliplr(lr_img)
					hr_img = np.fliplr(hr_img)
				lr_batch.append(lr_img)
				hr_batch.append(hr_img)

			# normalize input images to [-1, 1]
			lr_batch = (np.array(lr_batch) - 127.5) / 127.5
			hr_batch = (np.array(hr_batch) - 127.5) / 127.5
			queue.put([lr_batch, hr_batch])

	def calc_num_batches(self, train_step, final_training):
		num_batches_per_thread = 0
		if train_step and not final_training:
			num_batches_per_thread = int((self.num_training_images / self.BATCH_SIZE) // self.NUM_THREADS) // 6
		elif train_step: # and final_training:
			temp = int(((self.num_training_images / self.BATCH_SIZE) / self.NUM_THREADS) // (self.TRAINING_RATIO + 1))
			num_batches_per_thread = temp * (self.TRAINING_RATIO + 1)
		else: # for validation threads
			num_batches_per_thread = int((self.num_validation_images // self.BATCH_SIZE) // self.NUM_THREADS)

		num_batches_per_thread = 6
		num_batches = num_batches_per_thread * self.NUM_THREADS
		return num_batches_per_thread, num_batches

	def start_epoch_threads(self, data_gen_thread_func, file_path_list_lr, file_path_list_hr, train_step, final_training):
		queue_batches = JoinableQueue(self.MAX_QUEUE_SIZE)
		
		# calculate number of batches per data thread to generate
		# and overall number of batches
		self.num_batches_per_thread, self.num_batches = self.calc_num_batches(train_step=train_step, final_training=final_training)
		
		# start data generating threads
		list_threads = []
		for i in range(self.NUM_THREADS):
			worker = Process(target=data_gen_thread_func, args=(queue_batches, i, 
							file_path_list_lr, file_path_list_hr, train_step, final_training))
			worker.start()
			list_threads.append(worker)

		return queue_batches, list_threads

	def train_thread(self, file_path_train_imgs_lr, file_path_train_imgs_hr, file_path_valid_imgs_lr, file_path_valid_imgs_hr, final_training=False):
		file_path_train_list_lr = [os.path.join(file_path_train_imgs_lr, 'train', x) for x in os.listdir(os.path.join(file_path_train_imgs_lr, 'train'))]
		file_path_train_list_hr = [os.path.join(file_path_train_imgs_hr, 'train', x) for x in os.listdir(os.path.join(file_path_train_imgs_hr, 'train'))]
		file_path_valid_list_lr = [os.path.join(file_path_valid_imgs_lr, 'val', x) for x in os.listdir(os.path.join(file_path_valid_imgs_lr, 'val'))]
		file_path_valid_list_hr = [os.path.join(file_path_valid_imgs_hr, 'val', x) for x in os.listdir(os.path.join(file_path_valid_imgs_hr, 'val'))]

		if self.PRETRAIN_GENERATOR:
			self.best_psnr_pretrain = 0.0
			self.no_improvement_pretrain_epochs = 0
		
		self.best_discriminator_loss = -10.0
		self.no_improvement_wasserstein_loss = 0
		
		# validation results file
		csv_valid_epoch = open('results_srwgangp_valid_epoch.csv', 'a')
		valid_title_row = "epoch valid_PSNR\n"
		csv_valid_epoch.write(valid_title_row)
		csv_valid_epoch.close()

		# final train results files
		csv_train_final = open('results_srwgangp_train.csv', 'a')
		train_title_row = "iteration discriminator_train_loss generator_train_loss generator_train_PSNR\n"
		csv_train_final.write(train_title_row)
		csv_train_final.close()
		
		csv_train_epoch_final = open('results_srwgangp_train_epoch.csv', 'a')
		epoch_title_row = "epoch discriminator_train_loss generator_train_loss generator_train_PSNR\n"
		csv_train_epoch_final.write(epoch_title_row)
		csv_train_epoch_final.close()
		
		file_path_test_list_32 = [os.path.join(self.TEST_32_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_32_DATA_GENERATOR_PATH, 'test'))]
		file_path_test_list_128 = [os.path.join(self.TEST_128_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_128_DATA_GENERATOR_PATH, 'test'))]

		self.curr_iter_train = 0
		NUM_EPOCHS = 1
		if(not final_training):
			NUM_EPOCHS = self.PRETRAIN_EPOCHS
		else:
			NUM_EPOCHS = self.NUM_EPOCHS

		for epoch in range(NUM_EPOCHS):
			print('Epoch: ', epoch)
			# train for epoch
			self.epoch_curr = epoch

			# random shuffle of training data
			rand_seed = random.randint(1, 10000)
			random.Random(rand_seed).shuffle(file_path_train_list_lr)
			random.Random(rand_seed).shuffle(file_path_train_list_hr)
			random.Random(rand_seed).shuffle(file_path_valid_list_lr)
			random.Random(rand_seed).shuffle(file_path_valid_list_hr)

			queue_batches, threads_list = self.start_epoch_threads(
											data_gen_thread_func=self.data_gen_thread,
											file_path_list_lr=file_path_train_list_lr,
											file_path_list_hr=file_path_train_list_hr,
											train_step=True,
											final_training=final_training)
			self.train_epoch_thread(queue_batches=queue_batches, epoch=epoch, final_training=final_training)
			for worker in threads_list:
				worker.join()
			#print('Joined data generator threads.')
			queue_batches.join()
			#print('Joined train thread.')

			# validate
			queue_batches, threads_list = self.start_epoch_threads(
											data_gen_thread_func=self.data_gen_thread,
											file_path_list_lr=file_path_valid_list_lr,
											file_path_list_hr=file_path_valid_list_hr,
											train_step=False,
											final_training=final_training)
			self.valid_epoch_thread(queue_batches=queue_batches, epoch=epoch, final_training=final_training)

			for worker in threads_list:
				worker.join()
			#print('Joined data generator threads.')
			queue_batches.join()
			#print('Joined validation thread.')
			
			dir_images = 'Images/SRWGANGP'
			dir_images_NN = 'Images/NN_128'
			dir_images_HR = 'Imgaes/HR'

			self.process_and_save_images(save_directory=dir_images, save_directory_nearest=dir_images_NN, save_directory_hr=dir_images_HR,
								file_path_test_list_lr=file_path_test_list_32, file_path_test_list_hr=file_path_test_list_128,
								save_hr=False, save_nearest_neighbour=False)

	def train(self):
		# 128x128 HR		
		self.curr_HR_size = self.LR_TARGET_SIZE[0] * 4
		HR_TARGET_SIZE = (self.curr_HR_size, self.curr_HR_size)
		self.generator_model_train, self.discriminator_model = self.create_model_architecture(
																HR_TARGET_SIZE = HR_TARGET_SIZE)

		if self.PRETRAIN_GENERATOR:
			self.train_thread(file_path_train_imgs_lr=self.TRAIN_32_DATA_GENERATOR_PATH,
						file_path_train_imgs_hr=self.TRAIN_128_DATA_GENERATOR_PATH,
						file_path_valid_imgs_lr=self.VALIDATION_32_DATA_GENERATOR_PATH, 
						file_path_valid_imgs_hr=self.VALIDATION_128_DATA_GENERATOR_PATH, 
						final_training=False)
		
		# load pretrained generator weights
		if self.LOAD_PRETRAINED_GENERATOR and not self.PRETRAIN_GENERATOR:
			self.load_pretrained_generator(self.generator_model_train, self.PRETRAINED_GENERATOR_MODEL_PATH)
		
		self.train_thread(file_path_train_imgs_lr=self.TRAIN_32_DATA_GENERATOR_PATH,
						file_path_train_imgs_hr=self.TRAIN_128_DATA_GENERATOR_PATH,
						file_path_valid_imgs_lr=self.VALIDATION_32_DATA_GENERATOR_PATH, 
						file_path_valid_imgs_hr=self.VALIDATION_128_DATA_GENERATOR_PATH, 
						final_training=True)

	# loads pretrained generator's weights
	def load_pretrained_weights(self, new_model, pretrained_model):
		for i, (_, _) in enumerate(zip(new_model.layers, pretrained_model.layers)):
			new_model.layers[i].set_weights(pretrained_model.layers[i].get_weights())

	# sanity check if all weights are loaded correctly		
	def check_if_loaded_weights(self, new_model, pretrained_model):
		weights_loaded = True
		for i, (gen, pre) in enumerate(zip(new_model.layers, pretrained_model.layers)):
			layers_gen = gen.get_weights()
			layers_pre = pre.get_weights()
			
			if layers_gen:
				for gen_weights, pre_weights in zip(layers_gen, layers_pre):
					if not np.array_equal(gen_weights, pre_weights):
						weights_loaded = False

		if weights_loaded:
			print('Weights loaded.')
		else:
			print('Weights not loaded.')

	def load_pretrained_generator(self, new_model, pretrained_gen_model_path):
		print('Loading pretrained generator weights.')
		pretrained_model = load_model(pretrained_gen_model_path, 
							custom_objects={'RandomWeightedAverage':RandomWeightedAverage}, compile=False)
		self.load_pretrained_weights(new_model, pretrained_model)
		self.check_if_loaded_weights(new_model, pretrained_model)

	# preprocess image input as expected by VGG19 network
	def preproces_vgg(self, x):
		# scale from [-1,1] to [0, 255]
		x += 1.
		x *= 127.5
		
		# RGB -> BGR assuming data_format is 'channels_last':
		x = x[..., ::-1]
		
		# apply Imagenet preprocessing: BGR mean
		mean = [103.939, 116.778, 123.68]
		_IMAGENET_MEAN = K.constant(-np.array(mean))
		x = K.bias_add(x, K.cast(_IMAGENET_MEAN, K.dtype(x)))
		return x

	def vgg_loss(self, y_true, y_pred):
		# load pretrained VGG
		vgg19 = VGG19(include_top=False,
					  input_shape=self.HR_SHAPE, 
					  weights='imagenet')
		vgg19.trainable = False
		for l in vgg19.layers:
			l.trainable = False
		
		# uses features from layer 'block5_conv4'
		features_extractor = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output)
		
		# compute features for predicted and true images
		features_pred = features_extractor(self.preproces_vgg(y_pred))
		features_true = features_extractor(self.preproces_vgg(y_true))
		
		# adding the scaling factor (to have similar values as with MSE within image space)
		return 0.006*K.mean(K.square(features_pred - features_true), axis=-1)

	# PSNR definition not based on tensors as before
	def PSNR_val(self, y_true, y_pred):
		return -10. * np.log(np.mean(np.square(y_pred - y_true))) / np.log(10.)

	def save_weights(self, epoch):
		discr_model_path = 'Weights/discr_srwgangp_model_' + \
		str(self.curr_HR_size) + '_epoch_' + str(epoch) + '.h5'
		generator_model_path = 'Weights/gen_srwgangp_model_' + \
		str(self.curr_HR_size) + '_epoch_' + str(epoch) + '.h5'
		discr_weights_path = 'Weights/discr_srwgangp_weights_' + \
		str(self.curr_HR_size) + '_epoch_' + str(epoch) + '.h5'
		generator_weights_path = 'Weights/gen_srwgangp_weights_' + \
		str(self.curr_HR_size) + '_epoch_' + str(epoch) + '.h5'
	
		self.discriminator_model.save(discr_model_path)
		self.discriminator_model.save_weights(discr_weights_path)
		self.generator_model_train.save(generator_model_path)
		self.generator_model_train.save_weights(generator_weights_path)

	def get_batch_test_imgs(self, file_path_list_lr, file_path_list_hr):
		lr_batch = []
		hr_batch = []
		for index in range(self.BATCH_SIZE):
			lr_img = cv2.imread(file_path_list_lr[index], cv2.COLOR_BGR2RGB)
			hr_img = cv2.imread(file_path_list_hr[index], cv2.COLOR_BGR2RGB)

			lr_batch.append(lr_img)
			hr_batch.append(hr_img)

		# normalize input images to [-1, 1]
		lr_batch = (np.array(lr_batch) - 127.5) / 127.5
		hr_batch = (np.array(hr_batch) - 127.5) / 127.5

		return lr_batch, hr_batch

	# PSNR on test batch
	# returns an array of PSNR values
	# used to calculate batch psnr
	def PSNR_test_array(self, y_true, y_pred):
		psnr = -10. * np.log(np.mean(np.square(y_pred - y_true), (1, 2, 3))) / np.log(10.)
		return psnr

	# convert from [-1, 1] to [0, 255] values
	def denormalize_batch_images(self, image_batch):
		print(image_batch[0].shape)
		image_batch = [(img + 1.) * 127.5 for img in image_batch]
		return image_batch

	# save a batch of images
	def save_batch_images(self, directory, psnr_batch, image_batch):
		for i in range(len(image_batch)):
			filename = str(i) + '_e' + str(self.epoch_curr) +  '_' + str(psnr_batch[i]) + '.png'
			file_dir = os.path.join(directory, filename)
			cv2.imwrite(file_dir, image_batch[i])

	def process_and_save_images(self, save_directory, save_directory_nearest, save_directory_hr, file_path_test_list_lr, file_path_test_list_hr, 
								save_nearest_neighbour=False, save_hr=False):
		batch_test_32, batch_test_128 = self.get_batch_test_imgs(file_path_test_list_lr, file_path_test_list_hr)

		predicted_128 = self.generator_model_valid.predict_on_batch(batch_test_32)

		[_, predicted_128, _] = predicted_128

		PSNR_batch_test = self.PSNR_test_array(y_true=batch_test_128, y_pred=predicted_128)
		
		images_predicted = self.denormalize_batch_images(predicted_128)
		self.save_batch_images(directory=save_directory, psnr_batch=PSNR_batch_test, image_batch=images_predicted)
		
		if save_nearest_neighbour:
			batch_test_32 = self.denormalize_batch_images(batch_test_32)
			batch_128_nearest = []
			for i in range(self.BATCH_SIZE):
				img = misc.imresize(batch_test_32[i], NN_size, interp="nearest")
				batch_128_nearest.append(img)

			PSNR_batch_nearest = self.PSNR_test_array(y_true=batch_test_128, y_pred=batch_128_nearest)
			self.save_batch_images(directory=save_directory_nearest, psnr_batch=PSNR_batch_nearest, image_batch=batch_128_nearest)
	
		if save_hr:
			batch_test_denorm = self.denormalize_batch_images(batch_test_128)
			self.save_batch_images(directory=save_directory_hr, psnr_batch=PSNR_batch_test, image_batch=batch_test_denorm)

	def calc_psnr_test(self):
		file_path_test_list_lr = [os.path.join(self.TEST_32_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_32_DATA_GENERATOR_PATH, 'test'))]
		file_path_test_list_hr = [os.path.join(self.TEST_128_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_128_DATA_GENERATOR_PATH, 'test'))]

		final_training = False
		epoch=-1
		queue_batches, threads_list = self.start_epoch_threads(
										data_gen_thread_func=self.data_gen_thread,
										file_path_list_lr=file_path_test_list_lr,
										file_path_list_hr=file_path_test_list_hr,
										train_step=False,
										final_training=final_training)
		self.valid_epoch_thread(queue_batches=queue_batches, epoch=epoch, final_training=final_training)

		for worker in threads_list:
			worker.join()
		print('Joined data generator threads.')
		queue_batches.join()
		print('Joined test thread.')

	def mssim(self, y_true, y_pred):
		sum_ssim = 0.
		mssim_list = []
		for y_true_img, y_pred_img in zip(y_true, y_pred):
			calc_ssim = ssim(y_true_img, y_pred_img, data_range=2., multichannel=True)
			sum_ssim += calc_ssim
			mssim_list.append(calc_ssim)

		return np.mean(mssim_list)

	def create_images(self):
		file_path_test_list_32 = [os.path.join(self.TEST_32_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_32_DATA_GENERATOR_PATH, 'test'))]
		file_path_test_list_128 = [os.path.join(self.TEST_128_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_128_DATA_GENERATOR_PATH, 'test'))]
		file_path_test_list_256 = [os.path.join(self.TEST_256_DATA_GENERATOR_PATH, 'test', x) for x in os.listdir(os.path.join(self.TEST_256_DATA_GENERATOR_PATH, 'test'))]
		self.epoch_curr = -1

		self.curr_HR_size = self.LR_TARGET_SIZE[0] * 4
		HR_TARGET_SIZE = (self.curr_HR_size, self.curr_HR_size)
		self.generator_model_train, self.discriminator_model = self.create_model_architecture(
																HR_TARGET_SIZE = HR_TARGET_SIZE)
		# TODO SRWGANGP path
		gen_path_model = str('Weights/gen_srwgangp_model_128_epoch_0.h5')
		self.load_pretrained_generator(self.generator_model_train, gen_path_model)

		dir_images = 'Images/SRWGANGP'
		dir_images_NN = 'Images/NN_128'
		dir_images_HR = 'Images/HR'

		self.process_and_save_images(save_directory=dir_images, save_directory_nearest=dir_images_NN, save_directory_hr=dir_images_HR,
								file_path_test_list_lr=file_path_test_list_32, file_path_test_list_hr=file_path_test_list_128,
								save_hr=True, save_nearest_neighbour=True)
		print('SRWGANGP psnr test score:')
		self.calc_psnr_test()
								


if __name__ == '__main__':
	sr_wgan_gp = SRWGANGP()
	sr_wgan_gp.train()
	#sr_wgan_gp.create_images()
	