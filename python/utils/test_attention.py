import os
import sys
sys.path.append('../') # pycaffe module path
import caffe
import numpy as np
import cv2
from matplotlib import pyplot as plt

class net_wrapper() :
	"""
	caffe wrapper for test the network
	"""
	def __init__(self, deploy_prototxt, caffemodel, on_gpu=None) :
		if on_gpu is not None :
			caffe.set_mode_gpu()
			caffe.set_device(on_gpu)
		
		self._net = caffe.Net(deploy_prototxt, caffemodel, caffe.TEST)
		self._net_input = self._net.inputs[0]
		self._net_outputs = self._net.outputs
		self._num_class = (len(self._net_outputs)-1)/2
		self._input_dim = np.array( self._net.blobs[self._net_input].data.shape )
		self._mean = np.array([104,117,123]) # BGR order ...
		self._data_root = '/data/PASCAL/VOCdevkit/VOC2007/'
		# for evaluation
		self._dir_br_cnt = np.zeros( self._num_class ) 
		self._dir_tl_cnt = np.zeros( self._num_class ) 
		self._dir_total = np.zeros( self._num_class ) 
		self._cls_cnt = 0
		self._cls_total = 0

	def run_forward(self, patch_index) :
		batch = np.zeros( self._input_dim )
		for i in range(self._input_dim[0]) :
			batch[i] = self._make_patch_for_attention(patch_index+i)
		self._net.blobs[self._net_input].data[...] = batch
		self._output = self._net.forward()
		self._calc_accuracy(patch_index)
		
	def _calc_accuracy(self, patch_index) :
		# compare gt...
		for i in range(self._input_dim[0]) :
			gt_class = int(self._cls[patch_index+i])
			gt_TL = int(self._direction[patch_index+i][0])
			gt_BR = int(self._direction[patch_index+i][1])
		
			if gt_class != 20 :
				br_name = 'prob{}_BR'.format(gt_class)
				tl_name = 'prob{}_TL'.format(gt_class)
				pred_BR = self._output[br_name][i].squeeze().argmax()
				pred_TL = self._output[tl_name][i].squeeze().argmax()
				self._dir_total[gt_class] += 1
				if gt_TL == pred_TL :
					self._dir_tl_cnt[gt_class] += 1
				if gt_BR == pred_BR :
					self._dir_br_cnt[gt_class] += 1

			pred_cls = self._output['prob_cls'][i].squeeze().argmax()

			if gt_class == pred_cls :
				self._cls_cnt += 1
			self._cls_total += 1.
		
		#print 'gt: {}, {}, {}'.format(gt_class, gt_TL, gt_BR)	
		#print 'pr: {}, {}, {}'.format(pred_cls, pred_TL, pred_BR)
		#print '====================='	

	def _make_patch_for_attention(self, patch_index) :
		image_path = self._data_root + self._image_path[int(self._image_index[patch_index])]
		bbox = self._bbox[patch_index]
		img = cv2.imread(image_path)
		x1 = int(bbox[0])
		y1 = int(bbox[1])
		x2 = int(bbox[2])
		y2 = int(bbox[3])
		ih = img.shape[0]
		iw = img.shape[1]
		margin_l_x = 0 - x1
		margin_t_y = 0 - y1
		margin_r_x = x2 - iw + 1
		margin_b_y = y2 - ih + 1
		if ( margin_l_x > 0 or margin_t_y > 0 or margin_r_x > 0 or margin_b_y > 0 ) :
			if x1 < 0 : x1 = 0; 
			else : margin_l_x = 0;
			if y1 < 0 : y1 = 0; 
			else : margin_t_y = 0;
			if x2 >= iw : x2 = iw-1; 
			else : margin_r_x = 0;
			if y2 >= ih : y2 = ih-1; 
			else : margin_b_y = 0;
			patch = img[ y1:y2, x1:x2 ]
			patch = cv2.copyMakeBorder(patch,margin_t_y, margin_b_y, margin_l_x, margin_r_x, cv2.BORDER_CONSTANT, value=(self._mean))
			patch = cv2.resize(patch, (self._input_dim[2:][0],self._input_dim[2:][1]) )
		else :
			patch = img[ y1:y2, x1:x2 ]
			patch = cv2.resize(patch, (self._input_dim[2:][0],self._input_dim[2:][1]) )

		if int(self._flip[patch_index]) == 1 :
			patch = cv2.flip(patch,1)
		
		im = patch.transpose((2,0,1))
		im = im - self._mean[:,np.newaxis,np.newaxis]
		batch = np.zeros( (1, im.shape[0],self._input_dim[2:][0],self._input_dim[2:][1]) ) 
		batch[0] = im
		return batch

	def load_attention_val(self, txt_path) :
		self._image_path = []
		self._image_index = []
		self._bbox = []
		self._flip = []
		self._cls = []
		self._direction = []
		f = open(txt_path, 'r')
		while True :
			line = f.readline().split()
			if not line : break
			if line[0] is '#' :
				image_index = line[1]
				self._image_path.append( f.readline().split()[0] )
			if len(line) == 6+(self._num_class*2) : # bbox information
				self._image_index.append( image_index )
				self._bbox.append( [line[0],line[1],line[2],line[3]] )
				self._flip.append( line[4] )
				self._cls.append( line[-1] )
				cls = int(line[-1])
				if cls == self._num_class : # background
					self._direction.append( [4,4] )
				else :
					self._direction.append( [ line[5+2*cls], line[5+2*cls+1] ] )
		f.close()
		assert len(self._bbox) == len(self._flip) == len(self._cls) == len(self._image_index)

if __name__ == '__main__' :

	arch = 'bn_googlenet'
	setting = 'default_lr_0.01'
	
	prototxt_path = '/home/paeng/projects/2__ILSVRC/ilsvrc15/0__ATTENTION/0__MODELS/' + arch + '/' + setting + '/test.prototxt'
	model_path = '/home/paeng/projects/2__ILSVRC/ilsvrc15/0__ATTENTION/0__MODELS/' + arch + '/' + setting + '/ft_models/attention_voc_07_iter_19580.caffemodel'
	val_file_path = '/home/paeng/projects/2__ILSVRC/ilsvrc15/0__ATTENTION/1__DATA/PASCAL/val.txt'
	net = net_wrapper(prototxt_path, model_path, 0)
	batch_size = net._input_dim[0]
	net.load_attention_val(val_file_path)
	for i in xrange(0,len(net._image_index),batch_size) :
		net.run_forward(i)
		if i%3200 is 0 :
			print '{} th patch is evaluated... {}'.format(i,len(net._image_index))
			dir_acc = 0.
			for k in range(20) :
				if net._dir_total[k] != 0 :
					dir_acc += net._dir_br_cnt[k]/net._dir_total[k]
					dir_acc += net._dir_tl_cnt[k]/net._dir_total[k]
			print '        DIR_ERR: {:5f} %'.format( (1-(dir_acc/40.0))*100. )
			print '        CLS_ERR: {:5f} %'.format( (1-(net._cls_cnt/net._cls_total))*100. )

