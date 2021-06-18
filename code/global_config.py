from local_config import *

########################################################################
# 
# 			Training configs
# 
########################################################################

DEFAULT_BATCH_SIZE		= 12 if IS_LOCAL else 128
DEFAULT_NUM_WORKERS	 	= 2 if IS_LOCAL else 4
DEFAULT_MAX_EPOCH 		= 300
DEFAULT_PATIENCE		= 12

DEFAULT_M 				= 2 if IS_LOCAL else 40
DEFAULT_LR				= 0.001

GPU_COUNT 				= 0 if IS_LOCAL else 1

SGD_OPTIM 				= "sgd"
ADAM_OPTIM				= "adam"

DEFAULT_OPTIM 			= ADAM_OPTIM

OPTIM_CODES = [SGD_OPTIM, ADAM_OPTIM]

########################################################################
# 
# 			Dataset codes
# 
########################################################################

MNIST_CODE 			= "mnist"



DATASET_CODES = [MNIST_CODE]

NUM_CLASSES = {
	MNIST_CODE: 10,
}

########################################################################
# 
# 			Model codes
# 
########################################################################


DENSE_CODE			= "densenet"


DEFAULT_MODEL		= DENSE_CODE

MODEL_CODES = [DENSE_CODE]


########################################################################
# 
# 			Training codes
# 
########################################################################


SG_CODE					= "sg-mcmc"
	
DEFAULT_TRAINING_CODE   = SG_CODE

TRAINING_CODES = [SG_CODE]


########################################################################
# 
# 			Paths
# 
########################################################################



TB_LOGS_PATH		= ROOT_DIR + "tb_logs/"
CONFIG_PATH 		= CONFIG_DIR + "config.yml"

DATA_DIR 			= ROOT_DIR + "data/"

DATA_DIRS = dict()
for key in DATASET_CODES:
	DATA_DIRS[key] = DATA_DIR + key + "/"



########################################################################
# 
# 			Checkpoints
# 
########################################################################

FIXED_CKPTS = {
	"_mnist": "",
}




