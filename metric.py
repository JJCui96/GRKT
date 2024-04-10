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

def CONSIST(eval_paras):
	scores = torch.cat(eval_paras['consist_scores'], 0)[1:].mean()
	return scores.item()

def GAUC(eval_paras):
	scores = torch.cat(eval_paras['scores']).cpu().numpy()
	labels = torch.cat(eval_paras['labels']).cpu().numpy()
	probs = torch.cat(eval_paras['gauc_probs']).cpu().numpy()

	probs_scores = defaultdict(list)
	probs_labels = defaultdict(list)

	for i, prob in enumerate(probs):
		probs_scores[prob].append(scores[i])
		probs_labels[prob].append(labels[i])
	
	auc = list()
	length = list()

	for prob in probs_scores:

		prob_scores = np.array(probs_scores[prob])
		prob_labels = np.array(probs_labels[prob])
		if np.all(prob_labels == 1) or np.all(prob_labels == 0):
			continue 
		fpr, tpr, _ = metrics.roc_curve(prob_labels, prob_scores, pos_label = 1)
		auc.append(metrics.auc(fpr, tpr))
		length.append(len(probs_scores[prob]))
	
	return np.average(auc, weights = length)

def REPEAT(eval_paras):

	scores = torch.cat(eval_paras['repeat_scores']).cpu().numpy()
	labels = torch.cat(eval_paras['repeat_labels']).cpu().numpy()
	preds = scores > 0.5
	return metrics.accuracy_score(labels, preds)