
from __future__ import print_function
import unittest
import tempfile
import re
import os
import caffe
from caffe import layers as L
from caffe import params as P

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
	conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group,
								param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)] )
	return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks, stride=1):
	return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def make_vgg_prototxt_for_attention_net(file_name, num_classes, batch_size) :
	net = caffe.NetSpec()
	net.data, net.cls_label = L.AttentionData(root_folder="./", source="test", 
										  batch_size=32, num_class=2, input_size=224, mirror=1,
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
		pname = "loss%d_TL"%(i)
		tname = "dir%d_TL"%(i)
		lname = "dir%d_TL_label"%(i)
		net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname],
										loss_param=dict(attention_net_ignore_label=4) )
		pname = "loss%d_BR"%(i)
		tname = "dir%d_BR"%(i)
		lname = "dir%d_BR_label"%(i)
		net.tops[pname] = L.SoftmaxWithLoss(net.tops[tname], net.tops[lname],
											loss_param=dict(attention_net_ignore_label=4) )
	# classification loss layer
	net.loss_cls = L.SoftmaxWithLoss(net.cls, net.cls_label)
	
	with open(file_name, 'w') as f:
		print(net.to_proto(), file=f)

def make_vgg16_attention(proto_file_name) :
	num_classes = 2
	batch_size = 1
	file_name = "tmp_" + proto_file_name
	make_vgg_prototxt_for_attention_net(file_name, num_classes, batch_size)
	with open(file_name, 'r') as f :
		proto_str = f.read()
	layer_pattern = 'layer[\\s\\S]+?\\n}\\n'
	layers = re.findall(layer_pattern, proto_str)
	num_layer = len(layers)
	input_layer = layers[0]
	params = [k for k in input_layer.split('\n')]

	with open(proto_file_name, 'w') as f:
		for i in range(4) : 
			print(params[i], file=f)
		for i in range(num_classes) :
			lname = "dir%d_TL_label"%(i)
			f.write("  top: \"{}\"\n".format(lname))
			lname = "dir%d_BR_label"%(i)
			f.write("  top: \"{}\"\n".format(lname))
		for i in xrange(4, len(params)) : 
			print(params[i], file=f)

		for i in range(num_layer) :
			if i > 2*num_classes :
				print(layers[i], file=f)
	os.remove(file_name)

if __name__ == '__main__' :
	prototxt_file_name = "train-all.prototxt"
	make_vgg16_attention(prototxt_file_name)














