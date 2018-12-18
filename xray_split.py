from __future__ import print_function, division

import os
import argparse
import errno
import shutil

import torch
import numpy as np
import torchvision

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
import util
import pandas as pd
import remotedebugger as rd


parser = argparse.ArgumentParser(description='PyTorch script for chest xray view classification')
rd.attachDebugger(parser)
parser.add_argument('-o', '--output', help='output folder where classified images are stored', required=False)
parser.add_argument('-if', '--inputFile', help='input file with chest x-rays paths', required=False)
parser.add_argument('-i', '--input', help='input folder where chest x-rays are stored', required=False)

args = parser.parse_args()

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

def mkdir_p(path):
#function  by @tzot from stackoverflow
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


if args.output is not None:
	outputPath = args.output
else:
	outputPath = root + '/chestViewSplit/chest_xray/'

csvFile = True
inputFile = None
if args.inputFile is not None:
	inputFile  = root + '/Rx-thorax-automatic-captioning/' + args.inputFile
else:
	#inputFile = 'position_toreview_images.csv'
	inputFile = 'all_info_studies_labels_160K.csv'
	inputFile  = root + '/Rx-thorax-automatic-captioning/' + inputFile

if args.input is not None:
	csvFile = False
	root_front = os.path.join(args.output, 'front')
	root_side = os.path.join(args.output, 'side')

	mkdir_p(root_front)
	mkdir_p(root_side)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

views = {0: 'front',
		 1:	'side'}


data_transforms = {
	'test': transforms.Compose([
		transforms.Scale(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]),
}

dataset_dir = None
imgs =  None
df = None
if csvFile is False:
	dataset_dir = args.input
else:
	df = pd.read_csv(inputFile)
	imgs = df[df.Review == 'UNK']
	imgs['ImagePath'] = imgs['ImagePath'].apply(lambda x: root + '/SJ' + str(x) )
	imgs = imgs['ImagePath'].values

from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
print("| Loading chestViewNet for chest x-ray view classification...")
checkpoint = torch.load('./models/'+'resnet-50.t7',map_location=lambda storage, loc: storage, pickle_module=pickle)
model = checkpoint['model']

use_gpu = torch.cuda.is_available()
if use_gpu:
	model.cuda()
	print("Using cuda")


model.eval()
print("Past eval")

testsets = util.MyFolder(dataset_dir, imgs, data_transforms['test'], loader = util.SJ_loader)

testloader = torch.utils.data.DataLoader(
	testsets,
	batch_size = 1,
	shuffle = False,
	num_workers=1
)







print("\n| classifying ..." )
side = []
front = []
for batch_idx, (inputs, path) in enumerate(testloader):
	if use_gpu:
		inputs = inputs.cuda()
	inputs = Variable(inputs, volatile=True)
	outputs = model(inputs)

	softmax_res = softmax(outputs.data.cpu().numpy()[0])

	_, predicted = torch.max(outputs.data, 1)
	p = re.sub(r'.*/SJ', '', path[0])
	print('%s is %s view' % (p, views[predicted.cpu().numpy()[0]]))

	if predicted.cpu().numpy()[0] == 0:
		if csvFile is False:
			shutil.copy2(path[0], root_front)
		else:
			front.append(p)

	else:
		if csvFile is False:
			shutil.copy2( path[0], root_side)
		else:
			side.append( p)



pd.DataFrame(side).to_csv('position_side_predicted.csv')
pd.DataFrame(front).to_csv('position_front_predicted.csv')

if csvFile is True: 
	df.loc[df['ImagePath'].isin(side), 'Review'] = 'L'
	df.loc[df['ImagePath'].isin(front), 'Review'] = 'PA'
	df.to_csv('all_info_studies_labels_projections_160K.csv')
