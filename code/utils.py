import yaml


from global_config import *
import numpy as np



def get_model_desc(args, include_task_id=False):

	if isinstance(args, dict):
		training_code 		= args['training_code']
		dataset_code 		= args['dataset_code']
		model_code 			= args['model_code']
		task_id 			= args['task_id']

		if 'ckpt' in args:
			ckpt			= args['ckpt']

	else:
		training_code 		= args.training_code
		dataset_code 		= args.dataset_code
		model_code 			= args.model_code
		ckpt				= args.ckpt
		task_id 			= args.task_id

	name = f"{training_code}-{dataset_code}-{model_code}"
	if ckpt:
		name += f"-{ckpt}"

	if include_task_id:
		name += f"-{task_id}"

	return name


def override_from_config(args):

	path = CONFIG_PATH

	args_copy = args

	stream = open(path, 'r')
	cfg_dict = yaml.safe_load(stream)[args.task_id]
	args_dict = vars(args_copy)

	for key, value in cfg_dict.items():
		if key in args_dict:
			# set any which have been found in the yml
			args_dict[key] = value

	return args_copy