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
D_WEIGHT = 0.01
D_MULTIPLY = False
PATCH_SIZE = 64
MODELS_DIR = 'models_hednet_true_se_gan'
LOG_DIR = 'drlog_hednet_true_se_gan'
G_LEARNING_RATE = 0.001
D_LEARNING_RATE = 0.001
RESUME_MODEL = None
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = 'models_hednet_true_ma_gan/model_20.pth.tar'

#Modify the parameters for testing.
TEST_BATCH_SIZE = 2
TEST_MODEL = 'models_hednet_true_ma_gan/model_2680.pth.tar'
SAVE_OUTPUT_IMAGES = True
TEST_OUTPUT_DIR = 'test_true_pre_ma_gan'
