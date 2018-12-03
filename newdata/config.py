LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

#Modify the general parameters.
IMAGE_DIR = '/home/qiqix/SegmentationSub1'
LESION_NAME = 'SE'
CLASS_ID = LESION_IDS[LESION_NAME]
NET_NAME = 'hednet'
PREPROCESS = True
IMAGE_SIZE = 512

#Modify the parameters for training.
EPOCHES = 5000
TRAIN_BATCH_SIZE = 4
USE_DNET = False
D_WEIGHT = 100
D_MULTIPLY = False
PATCH_SIZE = 128
MODELS_DIR = 'models_hednet_true_se_gan'
LOG_DIR = 'drlog_hednet_true_se_gan'
LEARNING_RATE = 0.00025
RESUME_MODEL = None
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = 'models_hednet_true_se/model_2620.pth.tar'

#Modify the parameters for testing.
TEST_BATCH_SIZE = 2
TEST_MODEL = 'pretrained/model_1140_se.pth.tar'
SAVE_OUTPUT_IMAGES = True
TEST_OUTPUT_DIR = 'test_true_pre_se_gan'
