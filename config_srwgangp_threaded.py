class SRWGANGPConfig:
	HR_SHAPE = (128, 128, 3)
	HR_TARGET_SIZE = (128, 128)
	DOWNSCALE_FACTOR = 4
	LR_SHAPE = (HR_SHAPE[0] // DOWNSCALE_FACTOR, HR_SHAPE[1] // DOWNSCALE_FACTOR, 3)
	LR_TARGET_SIZE = (HR_TARGET_SIZE[0] // DOWNSCALE_FACTOR, HR_TARGET_SIZE[1] // DOWNSCALE_FACTOR)

	# whether to pretrain generator or not
	PRETRAIN_GENERATOR = True
	# whether to load pretrained generator weights
	LOAD_PRETRAINED_GENERATOR = False
	# pretrain generator model for PRETRAIN_EPOCHS, minimizing only MSE
	PRETRAIN_EPOCHS = 3

	# train whole model for NUM_EPOCHS
	NUM_EPOCHS = 50
	BATCH_SIZE = 32

	# number of data generator threads
	NUM_THREADS = 64
	# data generator queue size 
	MAX_QUEUE_SIZE = 80

	# decrease learning rate after LR_DECAY_NO_IMPROVEMENT_EPOCHS epochs
	# in case of pretrained generator: learning rate will decrease when PSNR value on validation set 
	# is not increased in consecutive LR_DECAY_NO_IMPROVEMENT_EPOCHS epochs
	# in case of normal training phase: generator's and discriminator's learning rate will decrease 
	# when discriminator's training loss is not improved in consecutive LR_DECAY_NO_IMPROVEMENT_EPOCHS epochs
	LR_DECAY_NO_IMPROVEMENT_EPOCHS = 2

	# decrease pretrained generator's learning rate by LR_DECAY_PRETRAIN
	LR_DECAY_PRETRAIN = 2.
	# decrease learning rate of generator and discriminator by LR_DECAY_FACTOR
	LR_DECAY_FACTOR = 2.

	# values from papers:
	# number of discriminator updates per generator update
	TRAINING_RATIO = 5
	# gradient penalty weight used in discriminator loss function
	GRADIENT_PENALTY_WEIGHT = 10

	# path to save and load a pretrained generator model
	PRETRAINED_GENERATOR_MODEL_PATH = 'Weights/pretrained_gen_model.h5'
	PRETRAINED_GENERATOR_WEIGHTS_PATH = 'Weights/pretrained_gen_weights.h5'

	# paths to training and validation images
	# they have to point to one directory above image files
	TRAIN_32_DATA_GENERATOR_PATH = 'facesDataset/faces/32/train_imgs'
	TRAIN_128_DATA_GENERATOR_PATH = 'facesDataset/faces/128/train_imgs'
	VALIDATION_32_DATA_GENERATOR_PATH = 'facesDataset/faces/32/val_imgs'
	VALIDATION_128_DATA_GENERATOR_PATH = 'facesDataset/faces/128/val_imgs'
	TEST_32_DATA_GENERATOR_PATH = 'facesDataset/faces/32/test_imgs'
	TEST_128_DATA_GENERATOR_PATH = 'facesDataset/faces/128/test_imgs'

	# paths to image folders
	LR_TRAIN_IMAGES_PATH = 'facesDataset/faces/32/train_imgs/train'
	LR_VALID_IMAGES_PATH = 'facesDataset/faces/32/val_imgs/val'
	LR_TEST_IMAGES_PATH  = 'facesDataset/faces/32/test_imgs/test'
