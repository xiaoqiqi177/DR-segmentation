LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

#Modify the general parameters.
IMAGE_DIR = '/home/qiqix/SegmentationSub1'
<<<<<<< HEAD
LESION_NAME = 'EX'
=======
LESION_NAME = 'SE'
>>>>>>> 18feeb6947c4d72a9f4f66a3fb9336e635f25a6e
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
<<<<<<< HEAD
MODELS_DIR = 'models_hednet_true_ex_gan'
LOG_DIR = 'drlog_hednet_true_ex_gan'
G_LEARNING_RATE = 0.001
D_LEARNING_RATE = 0.001
RESUME_MODEL = 'models_hednet_true_ex_gan/model_20.pth.tar'
=======
MODELS_DIR = 'models_hednet_true_se_gan'
LOG_DIR = 'drlog_hednet_true_se_gan'
G_LEARNING_RATE = 0.001
D_LEARNING_RATE = 0.001
RESUME_MODEL = None
>>>>>>> 18feeb6947c4d72a9f4f66a3fb9336e635f25a6e
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None

#Modify the parameters for testing.
TEST_BATCH_SIZE = 2
TEST_MODEL = 'pretrained/model_2640_ma.pth.tar'
SAVE_OUTPUT_IMAGES = True
TEST_OUTPUT_DIR = 'test_true_pre_ma_gan'
