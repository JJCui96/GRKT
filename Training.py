from models import *
from Runner import Runner
import argparse
import pickle
import sys

def training(argv = None):

	print('Training start...')
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--model', default = 'GRKT', type = str, 
		help = 'Model used.')
	parser.add_argument('--dataset', default = 'assist09', type = str, 
		help = 'Data used.')

	if argv:
		sys.argv = argv

	args, _ = parser.parse_known_args()
	model_class = eval('{0}.{0}'.format(args.model))
	with open('data/' + args.dataset + '/' + args.dataset + '.bin', 'rb') as f:
		dataset = pickle.load(f)

	Runner.parse_args(parser)
	model_class.parse_args(parser)

	args, _ = parser.parse_known_args()

	for key in dataset:
		if key != 'data':
			args.__setattr__(key, dataset[key])

	runner = Runner(args)
	result = runner.run(model_class, dataset['data'])
	return result


if __name__ == '__main__':
	training()
