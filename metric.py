from sklearn import metrics
import torch
from collections import defaultdict
import numpy as np

def AUC(eval_paras):
	scores = torch.cat(eval_paras['scores']).cpu()
	labels = torch.cat(eval_paras['labels']).cpu()

	fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label = 1)
	return metrics.auc(fpr, tpr)

def ACC(eval_paras):
	scores = torch.cat(eval_paras['scores']).cpu().numpy()
	labels = torch.cat(eval_paras['labels']).cpu().numpy()
	preds = scores > 0.5
	return metrics.accuracy_score(labels, preds)
