LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

#Modify the general parameters.
IMAGE_DIR = '/home/qiqix/SegmentationSub1'
LESION_NAME = 'EX'
CLASS_ID = LESION_IDS[LESION_NAME]
NET_NAME = 'hednet'
PREPROCESS = True
IMAGE_SIZE = 512

#Modify the parameters for testing.
TEST_BATCH_SIZE = 2
TEST_MODEL = 'models_hednet_true_ex_gan/model_4580.pth.tar'
SAVE_OUTPUT_IMAGES = True
TEST_OUTPUT_DIR = 'test_true_pre_ex_gan'
