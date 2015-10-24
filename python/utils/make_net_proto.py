
from __future__ import print_function
import unittest
import tempfile
import re
import os
import sys
sys.path.append('../') # pycaffe module path
import caffe
from caffe import layers as L
from caffe import params as P

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
	conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group,
		   param=[dict(lr_mult=0.25, decay_mult=1),dict(lr_mult=0.5, decay_mult=0)] )
	return conv, L.ReLU(conv, in_place=True)

def conv_bn_relu(bottom, ks, nout, stride=1, pad=0, group=1):
	conv = L.Convolution( bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, # bias_term=0,
		   param=[dict(lr_mult=0.25, decay_mult=1), dict(lr_mult=0.5, decay_mult=0)],
		   bias_filler=dict(type='constant',value=0) )
	bn = L.BN( conv, scale_filler=dict(type='constant',value=1), shift_filler=dict(type='constant',value=0),
		   param=[dict(lr_mult=0.25, decay_mult=0),dict(lr_mult=0.25, decay_mult=0)] )
	relu = L.ReLU( bn, in_place=True )
	return conv, bn, relu

def max_pool(bottom, ks, stride=1, pad=0):
	return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad)

def avg_pool(bottom, ks, stride=1, pad=0):
	return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride, pad=pad)

def bn_inception_pass(net, base_name, bottom, nout3r, nout3, noutd3r, noutd3):
	name3r = base_name + "/3x3_reduce"
	name3r_bn = base_name + "/3x3_reduce/bn"
	name3r_relu = base_name + "/relu_3x3_reduce"
	net.tops[name3r], net.tops[name3r_bn], net.tops[name3r_relu] = conv_bn_relu(bottom, 1, nout3r)
	name3 = base_name + "/3x3"
	name3_bn = base_name + "/3x3/bn"
	name3_relu = base_name + "/relu_3x3"
	net.tops[name3], net.tops[name3_bn], net.tops[name3_relu] = conv_bn_relu(net.tops[name3r_bn], 3, nout3, stride=2, pad=1)
	named3r = base_name + "/double3x3_reduce"
	named3r_bn = base_name + "/double3x3_reduce/bn"
	named3r_relu = base_name + "/relu_double3x3_reduce"
	net.tops[named3r], net.tops[named3r_bn], net.tops[named3r_relu] = conv_bn_relu(bottom, 1, noutd3r)
	named3a = base_name + "/double3x3a"
	named3a_bn = base_name + "/double3x3a/bn"
	named3a_relu = base_name + "/relu_double3x3a"
	net.tops[named3a], net.tops[named3a_bn], net.tops[named3a_relu] = conv_bn_relu(net.tops[named3r_bn], 3, noutd3, pad=1)
	named3b = base_name + "/double3x3b"
	named3b_bn = base_name + "/double3x3b/bn"
	named3b_relu = base_name + "/relu_double3x3b"
	net.tops[named3b], net.tops[named3b_bn], net.tops[named3b_relu] = conv_bn_relu(net.tops[named3a_bn], 3, noutd3, stride=2, pad=1)
	namep = base_name + "/pool/3x3_s2"
	net.tops[namep] = max_pool(bottom, 3, stride=2)
	nameo = base_name + "/output"
	net.tops[nameo] = L.Concat(net.tops[name3_bn],net.tops[named3b_bn],net.tops[namep])
	
	return nameo

def bn_inception(net, base_name, bottom, nout1, nout3r, nout3, noutd3r, noutd3, noutp, pool_method):
	name1 = base_name + "/1x1"
	name1_bn = base_name + "/1x1/bn"
	name1_relu = base_name + "/relu_1x1"
	net.tops[name1], net.tops[name1_bn], net.tops[name1_relu] = conv_bn_relu(bottom, 1, nout1)
	name3r = base_name + "/3x3_reduce"
	name3r_bn = base_name + "/3x3_reduce/bn"
	name3r_relu = base_name + "/relu_3x3_reduce"
	net.tops[name3r], net.tops[name3r_bn], net.tops[name3r_relu] = conv_bn_relu(bottom, 1, nout3r)
	name3 = base_name + "/3x3"
	name3_bn = base_name + "/3x3/bn"
	name3_relu = base_name + "/relu_3x3"
	net.tops[name3], net.tops[name3_bn], net.tops[name3_relu] = conv_bn_relu(net.tops[name3r_bn], 3, nout3, pad=1)
	named3r = base_name + "/double3x3_reduce"
	named3r_bn = base_name + "/double3x3_reduce/bn"
	named3r_relu = base_name + "/relu_double3x3_reduce"
	net.tops[named3r], net.tops[named3r_bn], net.tops[named3r_relu] = conv_bn_relu(bottom, 1, noutd3r)
	named3a = base_name + "/double3x3a"
	named3a_bn = base_name + "/double3x3a/bn"
	named3a_relu = base_name + "/relu_double3x3a"
	net.tops[named3a], net.tops[named3a_bn], net.tops[named3a_relu] = conv_bn_relu(net.tops[named3r_bn], 3, noutd3, pad=1)
	named3b = base_name + "/double3x3b"
	named3b_bn = base_name + "/double3x3b/bn"
	named3b_relu = base_name + "/relu_double3x3b"
	net.tops[named3b], net.tops[named3b_bn], net.tops[named3b_relu] = conv_bn_relu(net.tops[named3a_bn], 3, noutd3, pad=1)
	namep = base_name + "/pool"
	if pool_method is 'max' :
		net.tops[namep] = max_pool(bottom, 3, pad=1)
	elif pool_method is 'avg' :
		net.tops[namep] = avg_pool(bottom, 3, pad=1)
	namepp = base_name + "/pool_proj"
	namepp_bn = base_name + "/pool_proj/bn"
	namepp_relu = base_name + "/relu_pool_proj"
	net.tops[namepp], net.tops[namepp_bn], net.tops[namepp_relu] = conv_bn_relu(net.tops[namep], 1, noutp)
	nameo = base_name + "/output"
	net.tops[nameo] = L.Concat(net.tops[name1_bn],net.tops[name3_bn],net.tops[named3b_bn],net.tops[namepp_bn])
	
	return nameo

def inception(net, base_name, bottom, nout1, nout3r, nout3, nout5r, nout5, noutp):
	name1 = base_name + "/1x1"
	name1_relu = base_name + "/relu_1x1"
	net.tops[name1], net.tops[name1_relu] = conv_relu(bottom, 1, nout1)
	name3r = base_name + "/3x3_reduce"
	name3r_relu = base_name + "/relu_3x3_reduce"
	net.tops[name3r], net.tops[name3r_relu] = conv_relu(bottom, 1, nout3r)
	name3 = base_name + "/3x3"
	name3_relu = base_name + "/relu_3x3"
	net.tops[name3], net.tops[name3_relu] = conv_relu(net.tops[name3r], 3, nout3, pad=1)
	name5r = base_name + "/5x5_reduce"
	name5r_relu = base_name + "/relu_5x5_reduce"
	net.tops[name5r], net.tops[name5r_relu] = conv_relu(bottom, 1, nout5r)
	name5 = base_name + "/5x5"
	name5_relu = base_name + "/relu_5x5"
	net.tops[name5], net.tops[name5_relu] = conv_relu(net.tops[name5r], 5, nout5, pad=2)
	namep = base_name + "/pool"
	net.tops[namep] = max_pool(bottom, 3, pad=1)
	namepp = base_name + "/pool_proj"
	namepp_relu = base_name + "/relu_pool_proj"
	net.tops[namepp], net.tops[namepp_relu] = conv_relu(net.tops[namep], 1, noutp)
	nameo = base_name + "/output"
	net.tops[nameo] = L.Concat(net.tops[name1],net.tops[name3],net.tops[name5],net.tops[namepp])
	
	return nameo

def make_bn_googlenet_prototxt_for_attention_net(file_name, num_classes, batch_size, phase) :
	net = caffe.NetSpec()
	net.data, net.cls_label = L.AttentionData(root_folder="/data/PASCAL/VOCdevkit/VOC2007/", source="1__DATA/PASCAL/train.txt", 
							  batch_size=batch_size, num_class=num_classes, input_size=224, cache_images=0,
							  mean_value=[104,117,123], ntop=2)

	for i in range(num_classes) :
		lname = "dir%d_TL_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
		lname = "dir%d_BR_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
	
	# net start !!
	cname1 = "conv1/7x7_s2"
	bname1 = "conv1/7x7_s2/bn"
	rname1 = "conv1/relu_7x7"
	net.tops[cname1], net.tops[bname1], net.tops[rname1] = conv_bn_relu(net.data, 7, 64, stride=2, pad=3 )
	pname1 = "pool1/3x3_s2"
	net.tops[pname1] = max_pool(net.tops[bname1], 3, stride=2)
	cname2_1 = "conv2/3x3_reduce"
	bname2_1 = "conv2/3x3_reduce/bn"
	rname2_1 = "conv2/relu_3x3_reduce"
	net.tops[cname2_1], net.tops[bname2_1], net.tops[rname2_1] = conv_bn_relu(net.tops[pname1], 1, 64)
	cname2_2 = "conv2/3x3"
	bname2_2 = "conv2/3x3/bn"
	rname2_2 = "conv2/relu_3x3"
	net.tops[cname2_2], net.tops[bname2_2], net.tops[rname2_2] = conv_bn_relu(net.tops[bname2_1], 3, 192, pad=1)
	pname2 = "pool2/3x3_s2"
	net.tops[pname2] = max_pool(net.tops[bname2_2], 3, stride=2)
	# inception start !!
	out_name = bn_inception(net, "inception_3a", net.tops[pname2], 64, 64, 64, 64, 96, 32, 'avg')
	out_name = bn_inception(net, "inception_3b", net.tops[out_name], 64, 64, 96, 64, 96, 64, 'avg')
	out_name = bn_inception_pass(net, "inception_3c", net.tops[out_name], 128, 160, 64, 96)
	out_name = bn_inception(net, "inception_4a", net.tops[out_name], 224, 64, 96, 96, 128, 128, 'avg')
	out_name = bn_inception(net, "inception_4b", net.tops[out_name], 192, 96, 128, 96, 128, 128, 'avg')
	out_name = bn_inception(net, "inception_4c", net.tops[out_name], 160, 128, 160, 128, 160, 128, 'avg')
	out_name = bn_inception(net, "inception_4d", net.tops[out_name], 96, 128, 192, 160, 192, 128, 'avg')
	out_name = bn_inception_pass(net, "inception_4e", net.tops[out_name], 128, 192, 192, 256)
	out_name = bn_inception(net, "inception_5a", net.tops[out_name], 352, 192, 320, 160, 224, 128, 'avg')
	out_name = bn_inception(net, "inception_5b", net.tops[out_name], 352, 192, 320, 192, 224, 128, 'max')
	pname5 = "pool5/7x7_s1"
	net.tops[pname5] = avg_pool(net.tops[out_name], 7)

	# final layer creation
	for i in range(num_classes) :
		tname = "dir%d_TL"%(i)
		net.tops[tname] = L.Convolution(net.tops[pname5], kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
					      weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
		tname = "dir%d_BR"%(i)
		net.tops[tname] = L.Convolution(net.tops[pname5], kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
						  weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
	# final classification layer
	net.cls = L.Convolution(net.tops[pname5], kernel_size=1, num_output=num_classes+1, 
              param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
			  weight_filler=dict(type='gaussian',std=0.01),
			  bias_filler=dict(type='constant',value=0) )
	if phase is 'TRAIN' :
		# loss layer creation
		for i in range(num_classes) :
			cname = "acc%d_TL"%(i)
			pname = "loss%d_TL"%(i)
			tname = "dir%d_TL"%(i)
			lname = "dir%d_TL_label"%(i)
			net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=1./3.,
							  loss_param=dict(attention_net_ignore_label=4) )
			net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
							  include=dict(phase=1))
			cname = "acc%d_BR"%(i)
			pname = "loss%d_BR"%(i)
			tname = "dir%d_BR"%(i)
			lname = "dir%d_BR_label"%(i)
			net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=1./3.,
							  loss_param=dict(attention_net_ignore_label=4) )
			net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
							  include=dict(phase=1))
		# classification loss layer
		net.loss_cls = L.SoftmaxWithLoss(net.cls, net.cls_label, loss_weight=1./3., 
					   loss_param=dict(attention_net_ignore_label=-1) )
		net.acc_cls = L.Accuracy(net.cls, net.cls_label, attention_net_ignore_label=-1,
					  include=dict(phase=1))
	elif phase is 'TEST' :
		for i in range(num_classes) :
			pname = "prob%d_TL"%(i)
			tname = "dir%d_TL"%(i)
			net.tops[pname] = L.Softmax(net.tops[tname])
			pname = "prob%d_BR"%(i)
			tname = "dir%d_BR"%(i)
			net.tops[pname] = L.Softmax(net.tops[tname])
		# classification loss layer
		net.prob_cls = L.Softmax(net.cls)
	# save prototxt file
	with open(file_name, 'w') as f:
		print(net.to_proto(), file=f)

def make_googlenet_more_nonlinear_prototxt_for_attention_net(file_name, num_classes, batch_size, phase) :
	net = caffe.NetSpec()
	net.data, net.cls_label = L.AttentionData(root_folder="/data/PASCAL/VOCdevkit/VOC2007/", source="1__DATA/PASCAL/train.txt", 
							  batch_size=batch_size, num_class=num_classes, input_size=224, cache_images=0,
							  mean_value=[104,117,123], ntop=2)

	for i in range(num_classes) :
		lname = "dir%d_TL_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
		lname = "dir%d_BR_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
	
	# net start !!
	cname1 = "conv1/7x7_s2"
	rname1 = "conv1/relu_7x7"
	net.tops[cname1], net.tops[rname1] = conv_relu(net.data, 7, 64, stride=2, pad=3)
	pname1 = "pool1/3x3_s2"
	net.tops[pname1] = max_pool(net.tops[cname1], 3, stride=2)
	cname2_1 = "conv2/3x3_reduce"
	rname2_1 = "conv2/relu_3x3_reduce"
	net.tops[cname2_1], net.tops[rname2_1] = conv_relu(net.tops[pname1], 1, 64)
	cname2_2 = "conv2/3x3"
	rname2_2 = "conv2/relu_3x3"
	net.tops[cname2_2], net.tops[rname2_2] = conv_relu(net.tops[cname2_1], 3, 192, pad=1)
	pname2 = "pool2/3x3_s2"
	net.tops[pname2] = max_pool(net.tops[cname2_2], 3, stride=2)
	# inception start !!
	out_name = inception(net, "inception_3a", net.tops[pname2], 64, 96, 128, 16, 32, 32)
	out_name = inception(net, "inception_3b", net.tops[out_name], 128, 128, 192, 32, 96, 64)
	pname3 = "pool3/3x3_s2"
	net.tops[pname3] = max_pool(net.tops[out_name], 3, stride=2)
	out_name = inception(net, "inception_4a", net.tops[pname3], 192, 96, 208, 16, 48, 64)
	out_name = inception(net, "inception_4b", net.tops[out_name], 160, 112, 224, 24, 64, 64)
	out_name = inception(net, "inception_4c", net.tops[out_name], 128, 128, 256, 24, 64, 64)
	out_name = inception(net, "inception_4d", net.tops[out_name], 112, 144, 288, 32, 64, 64)
	out_name = inception(net, "inception_4e", net.tops[out_name], 256, 160, 320, 32, 128, 128)
	pname4 = "pool4/3x3_s2"
	net.tops[pname4] = max_pool(net.tops[out_name], 3, stride=2)
	out_name = inception(net, "inception_5a", net.tops[pname4], 256, 160, 320, 32, 128, 128)
	out_name = inception(net, "inception_5b", net.tops[out_name], 384, 192, 384, 48, 128, 128)
	pname5 = "pool5/7x7_s1"
	net.tops[pname5] = avg_pool(net.tops[out_name], 7)
	pname5_drop ="pool5/drop_7x7_s1"
	net.tops[pname5_drop] = L.Dropout(net.tops[pname5], in_place=True)
	
	# three more non-linearity for direction loss
	cname6 = "conv6"
	rname6 = "conv6_relu"
	net.tops[cname6], net.tops[rname6] = conv_relu(net.tops[pname5], 1, 512)
	
	# final layer creation
	for i in range(num_classes) :
		tname = "dir%d_TL"%(i)
		net.tops[tname] = L.Convolution(net.tops[cname6], kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
					      weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
		tname = "dir%d_BR"%(i)
		net.tops[tname] = L.Convolution(net.tops[cname6], kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
						  weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
	# final classification layer
	net.cls = L.Convolution(net.tops[pname5], kernel_size=1, num_output=num_classes+1, 
              param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
			  weight_filler=dict(type='gaussian',std=0.01),
			  bias_filler=dict(type='constant',value=0) )

	if phase is 'TRAIN' :
		# loss layer creation
		for i in range(num_classes) :
			cname = "acc%d_TL"%(i)
			pname = "loss%d_TL"%(i)
			tname = "dir%d_TL"%(i)
			lname = "dir%d_TL_label"%(i)
			net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=1./3.,
							  loss_param=dict(attention_net_ignore_label=4) )
			net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
							  include=dict(phase=1))
			cname = "acc%d_BR"%(i)
			pname = "loss%d_BR"%(i)
			tname = "dir%d_BR"%(i)
			lname = "dir%d_BR_label"%(i)
			net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=1./3.,
							  loss_param=dict(attention_net_ignore_label=4) )
			net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
							  include=dict(phase=1))
		# classification loss layer
		net.loss_cls = L.SoftmaxWithLoss(net.cls, net.cls_label, loss_weight=1./3., 
					   loss_param=dict(attention_net_ignore_label=-1) )
		net.acc_cls = L.Accuracy(net.cls, net.cls_label, attention_net_ignore_label=-1,
					  include=dict(phase=1))
	elif phase is 'TEST' :
		for i in range(num_classes) :
			pname = "prob%d_TL"%(i)
			tname = "dir%d_TL"%(i)
			net.tops[pname] = L.Softmax(net.tops[tname])
			pname = "prob%d_BR"%(i)
			tname = "dir%d_BR"%(i)
			net.tops[pname] = L.Softmax(net.tops[tname])
		# classification loss layer
		net.prob_cls = L.Softmax(net.cls)
	
	# save prototxt file
	with open(file_name, 'w') as f:
		print(net.to_proto(), file=f)

def make_googlenet_prototxt_for_attention_net(file_name, num_classes, batch_size, phase) :
	net = caffe.NetSpec()
	net.data, net.cls_label = L.AttentionData(root_folder="/data/PASCAL/VOCdevkit/VOC2007/", source="1__DATA/PASCAL/train.txt", 
							  batch_size=batch_size, num_class=num_classes, input_size=224, cache_images=0,
							  mean_value=[104,117,123], ntop=2)

	for i in range(num_classes) :
		lname = "dir%d_TL_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
		lname = "dir%d_BR_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
	
	# net start !!
	cname1 = "conv1/7x7_s2"
	rname1 = "conv1/relu_7x7"
	net.tops[cname1], net.tops[rname1] = conv_relu(net.data, 7, 64, stride=2, pad=3)
	pname1 = "pool1/3x3_s2"
	net.tops[pname1] = max_pool(net.tops[cname1], 3, stride=2)
	cname2_1 = "conv2/3x3_reduce"
	rname2_1 = "conv2/relu_3x3_reduce"
	net.tops[cname2_1], net.tops[rname2_1] = conv_relu(net.tops[pname1], 1, 64)
	cname2_2 = "conv2/3x3"
	rname2_2 = "conv2/relu_3x3"
	net.tops[cname2_2], net.tops[rname2_2] = conv_relu(net.tops[cname2_1], 3, 192, pad=1)
	pname2 = "pool2/3x3_s2"
	net.tops[pname2] = max_pool(net.tops[cname2_2], 3, stride=2)
	# inception start !!
	out_name = inception(net, "inception_3a", net.tops[pname2], 64, 96, 128, 16, 32, 32)
	out_name = inception(net, "inception_3b", net.tops[out_name], 128, 128, 192, 32, 96, 64)
	pname3 = "pool3/3x3_s2"
	net.tops[pname3] = max_pool(net.tops[out_name], 3, stride=2)
	out_name = inception(net, "inception_4a", net.tops[pname3], 192, 96, 208, 16, 48, 64)
	out_name = inception(net, "inception_4b", net.tops[out_name], 160, 112, 224, 24, 64, 64)
	out_name = inception(net, "inception_4c", net.tops[out_name], 128, 128, 256, 24, 64, 64)
	out_name = inception(net, "inception_4d", net.tops[out_name], 112, 144, 288, 32, 64, 64)
	out_name = inception(net, "inception_4e", net.tops[out_name], 256, 160, 320, 32, 128, 128)
	pname4 = "pool4/3x3_s2"
	net.tops[pname4] = max_pool(net.tops[out_name], 3, stride=2)
	out_name = inception(net, "inception_5a", net.tops[pname4], 256, 160, 320, 32, 128, 128)
	out_name = inception(net, "inception_5b", net.tops[out_name], 384, 192, 384, 48, 128, 128)
	pname5 = "pool5/7x7_s1"
	net.tops[pname5] = avg_pool(net.tops[out_name], 7)
	pname5_drop ="pool5/drop_7x7_s1"
	net.tops[pname5_drop] = L.Dropout(net.tops[pname5], in_place=True)

	# final layer creation
	for i in range(num_classes) :
		tname = "dir%d_TL"%(i)
		net.tops[tname] = L.Convolution(net.tops[pname5], kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
					      weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
		tname = "dir%d_BR"%(i)
		net.tops[tname] = L.Convolution(net.tops[pname5], kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
						  weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
	# final classification layer
	net.cls = L.Convolution(net.tops[pname5], kernel_size=1, num_output=num_classes+1, 
              param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
			  weight_filler=dict(type='gaussian',std=0.01),
			  bias_filler=dict(type='constant',value=0) )

	if phase is 'TRAIN' :
		# loss layer creation
		for i in range(num_classes) :
			cname = "acc%d_TL"%(i)
			pname = "loss%d_TL"%(i)
			tname = "dir%d_TL"%(i)
			lname = "dir%d_TL_label"%(i)
			net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=1./3.,
							  loss_param=dict(attention_net_ignore_label=4) )
			net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
							  include=dict(phase=1))
			cname = "acc%d_BR"%(i)
			pname = "loss%d_BR"%(i)
			tname = "dir%d_BR"%(i)
			lname = "dir%d_BR_label"%(i)
			net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=1./3.,
							  loss_param=dict(attention_net_ignore_label=4) )
			net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
							  include=dict(phase=1))
		# classification loss layer
		net.loss_cls = L.SoftmaxWithLoss(net.cls, net.cls_label, loss_weight=1./3., 
					   loss_param=dict(attention_net_ignore_label=-1) )
		net.acc_cls = L.Accuracy(net.cls, net.cls_label, attention_net_ignore_label=-1,
					  include=dict(phase=1))
	elif phase is 'TEST' :
		for i in range(num_classes) :
			pname = "prob%d_TL"%(i)
			tname = "dir%d_TL"%(i)
			net.tops[pname] = L.Softmax(net.tops[tname])
			pname = "prob%d_BR"%(i)
			tname = "dir%d_BR"%(i)
			net.tops[pname] = L.Softmax(net.tops[tname])
		# classification loss layer
		net.prob_cls = L.Softmax(net.cls)
	
	# save prototxt file
	with open(file_name, 'w') as f:
		print(net.to_proto(), file=f)

def make_vgg_prototxt_for_attention_net(file_name, num_classes, batch_size) :
	net = caffe.NetSpec()
	net.data, net.cls_label = L.AttentionData(root_folder="/data/PASCAL/VOCdevkit/VOC2007/", source="1__DATA/PASCAL/train.txt", 
							  batch_size=32, num_class=num_classes, input_size=224, cache_images=0,
							  mean_value=[104,117,123], ntop=2)
	for i in range(num_classes) :
		lname = "dir%d_TL_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
		lname = "dir%d_BR_label"%(i)
		net.tops[lname] = L.ReLU(net.data, in_place=False)
	
	# net start !!
	net.conv1_1, net.relu1_1 = conv_relu(net.data, 3, 64, pad=1)
	net.conv1_2, net.relu1_2 = conv_relu(net.conv1_1, 3, 64, pad=1)
	net.pool1 = max_pool(net.conv1_2, 2, stride=2)
	
	net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 3, 128, pad=1)
	net.conv2_2, net.relu2_2 = conv_relu(net.conv2_1, 3, 128, pad=1)
	net.pool2 = max_pool(net.conv2_2, 2, stride=2)
	
	net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 3, 256, pad=1)
	net.conv3_2, net.relu3_2 = conv_relu(net.conv3_1, 3, 256, pad=1)
	net.conv3_3, net.relu3_3 = conv_relu(net.conv3_2, 3, 256, pad=1)
	net.pool3 = max_pool(net.conv3_3, 2, stride=2)

	net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 3, 512, pad=1)
	net.conv4_2, net.relu4_2 = conv_relu(net.conv4_1, 3, 512, pad=1)
	net.conv4_3, net.relu4_3 = conv_relu(net.conv4_2, 3, 512, pad=1)
	net.pool4 = max_pool(net.conv4_3, 2, stride=2)
	
	net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 3, 512, pad=1)
	net.conv5_2, net.relu5_2 = conv_relu(net.conv5_1, 3, 512, pad=1)
	net.conv5_3, net.relu5_3 = conv_relu(net.conv5_2, 3, 512, pad=1)
	net.pool5 = max_pool(net.conv5_3, 2, stride=2)
	
	net.fc6_conv, net.relu6 = conv_relu(net.pool5, 7, 4096)
	net.drop6 = L.Dropout(net.relu6, in_place=True)
	
	net.fc7_conv, net.relu7 = conv_relu(net.drop6, 1, 4096)
	net.drop7 = L.Dropout(net.relu7, in_place=True)

	# final layer creation
	for i in range(num_classes) :
		tname = "dir%d_TL"%(i)
		net.tops[tname] = L.Convolution(net.drop7, kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
					      weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
		tname = "dir%d_BR"%(i)
		net.tops[tname] = L.Convolution(net.drop7, kernel_size=1, num_output=4, 
                          param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
						  weight_filler=dict(type='gaussian',std=0.01),
						  bias_filler=dict(type='constant',value=0) )
	# final classification layer
	net.cls = L.Convolution(net.drop7, kernel_size=1, num_output=num_classes+1, 
              param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
			  weight_filler=dict(type='gaussian',std=0.01),
			  bias_filler=dict(type='constant',value=0) )
	# loss layer creation
	for i in range(num_classes) :
		cname = "acc%d_TL"%(i)
		pname = "loss%d_TL"%(i)
		tname = "dir%d_TL"%(i)
		lname = "dir%d_TL_label"%(i)
		net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=0.33,
						  loss_param=dict(attention_net_ignore_label=4) )
		net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
						  include=dict(phase=1))
		cname = "acc%d_BR"%(i)
		pname = "loss%d_BR"%(i)
		tname = "dir%d_BR"%(i)
		lname = "dir%d_BR_label"%(i)
		net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname], loss_weight=0.33,
						  loss_param=dict(attention_net_ignore_label=4) )
		net.tops[cname] = L.Accuracy(net.tops[tname], net.tops[lname], attention_net_ignore_label=4,
						  include=dict(phase=1))
	# classification loss layer
	net.loss_cls = L.SoftmaxWithLoss(net.cls, net.cls_label, loss_weight=0.33, 
				   loss_param=dict(attention_net_ignore_label=-1) )
	net.acc_cls = L.Accuracy(net.cls, net.cls_label, attention_net_ignore_label=-1,
				   include=dict(phase=1))
	# save prototxt file
	with open(file_name, 'w') as f:
		print(net.to_proto(), file=f)

def make_attention_prototxt(proto_file_name, model, phase='TRAIN') :
	num_classes = 200
	batch_size = 64
	file_name = "tmp_" + proto_file_name
	
	if model is 'vgg16':
		make_vgg_prototxt_for_attention_net(file_name, num_classes, batch_size)
	elif model is 'bvlc_googlenet' :
		make_googlenet_prototxt_for_attention_net(file_name, num_classes, batch_size, phase)
	elif model is 'bvlc_googlenet_more_nonlinear' :
		make_googlenet_more_nonlinear_prototxt_for_attention_net(file_name, num_classes, batch_size, phase)
	elif model is 'googlenet_bn' :
		make_bn_googlenet_prototxt_for_attention_net(file_name, num_classes, batch_size, phase)
	
	with open(file_name, 'r') as f :
		proto_str = f.read()
	layer_pattern = 'layer[\\s\\S]+?\\n}\\n'
	layers = re.findall(layer_pattern, proto_str)
	num_layer = len(layers)
	input_layer = layers[0]
	params = [k for k in input_layer.split('\n')]

	with open(proto_file_name, 'w') as f:
		if phase is 'TRAIN' :
			for s in range(2) :
				for i in range(4) : 
					print(params[i], file=f)
				for i in range(num_classes) :
					lname = "dir%d_TL_label"%(i)
					f.write("  top: \"{}\"\n".format(lname))
					lname = "dir%d_BR_label"%(i)
					f.write("  top: \"{}\"\n".format(lname))
				print(params[4], file=f)
				if s == 0 :
					f.write("  include {\n    phase: TRAIN\n  }\n")
				else :
					f.write("  include {\n    phase: TEST\n  }\n")
				for i in xrange(5, len(params)) : 
					print(params[i], file=f)

		for i in range(num_layer) :
			if i > 2*num_classes :
				print(layers[i], file=f)
	os.remove(file_name)


if __name__ == '__main__' :
	prototxt_file_name = "train.prototxt"
	make_attention_prototxt(prototxt_file_name, 'bvlc_googlenet', 'TRAIN')
	#make_attention_prototxt(prototxt_file_name, 'googlenet_bn', 'TEST')

