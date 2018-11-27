LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

#Modify the general parameters.
IMAGE_DIR = '/home/qiqix/SegmentationSub1'
LESION_NAME = 'MA'
CLASS_ID = LESION_IDS[LESION_NAME]
NET_NAME = 'hednet'
PREPROCESS = True
IMAGE_SIZE = 512

#Modify the parameters for training.
EPOCHES = 5000
TRAIN_BATCH_SIZE = 4
D_WEIGHT = 0.1
D_MULTIPLY = False
PATCH_SIZE = 32
MODELS_DIR = 'models_hednet_true_ma_gan'
LOG_DIR = 'drlog_hednet_true_ma_gan'
LEARNING_RATE = 0.1
RESUME_MODEL = None
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None

#Modify the parameters for testing.
TEST_BATCH_SIZE = 2
TEST_MODEL = 'pretrained/model_2640_ma.pth.tar'
SAVE_OUTPUT_IMAGES = True
TEST_OUTPUT_DIR = 'test_true_pre_ma_gan'
