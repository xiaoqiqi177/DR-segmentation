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
PATCH_SIZE = 128
MODELS_DIR = 'results/models_hednet_true_se_gan'
LOG_DIR = 'drlog_hednet_true_se_gan'
G_LEARNING_RATE = 0.001
D_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = 'results/models_hednet_true_se_gan/model_500.pth.tar'
