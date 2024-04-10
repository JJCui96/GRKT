import torch
import random
import numpy as np

def feature2tensor(batch, pos, tensor_type, dim = 2, pad = 0):

	if dim == 1:
		arrays = [r[pos] for r in batch]
	elif dim == 1.5:
		max_feat_len = max([len(r[pos]) for r in batch])
		arrays = [r[pos] + [pad]*(max_feat_len - len(r[pos])) for r in batch]
	elif dim == 2:
		max_len = max([len(seq) for seq in batch])
		arrays = [[r[pos] for r in seq] + [pad]*(max_len - len(seq)) for seq in batch]
	elif dim == 3:
		max_len = max([len(seq) for seq in batch])
		max_feat_len = max([max([len(r[pos]) for r in seq]) for seq in batch])
		arrays = [[r[pos] + [0]*(max_feat_len - len(r[pos])) for r in seq] \
			+ ([[0]*max_feat_len])*(max_len - len(seq)) for seq in batch]

	if tensor_type == 'int':	return torch.LongTensor(arrays)
	if tensor_type == 'float':	return torch.FloatTensor(arrays)
	if tensor_type == 'bool':	return torch.BoolTensor(arrays)

def get_performance(eval_dict):
	res = 1.0
	for key in eval_dict:
		res *= eval_dict[key]
	res = res**(1/len(eval_dict))
	return res

def performance_str(eval_dict):
	log = ''
	for key in eval_dict:
		log += '{}: {:.4f}, '.format(key, eval_dict[key])
	return log
				
def reset_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True