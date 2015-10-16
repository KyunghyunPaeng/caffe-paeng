import sys
import numpy
import re
import pylab as pl
from collections import OrderedDict

def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, learning_rate):
	"""Parse a single line for training or test output

	Returns a a tuple with (row_dict_list, row)
	row: may be either a new row or an augmented version of the current row
	row_dict_list: may be either the current row_dict_list or an augmented
	version of the current row_dict_list
	"""
	output_match = regex_obj.search(line)
	if output_match:
		if not row or row['NumIters'] != iteration:
			# Push the last row and start a new one
			if row:
				# If we're on a new iteration, push the last row
				# This will probably only happen for the first row; otherwise
				# the full row checking logic below will push and clear full
				# rows
				row_dict_list.append(row)
			row = OrderedDict( [('NumIters', iteration), ('LearningRate', learning_rate)] )
		output_name = output_match.group(2)
		output_val = output_match.group(3)
		row[output_name] = float(output_val)
	if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
		# The row is full, based on the fact that it has the same number of
		# columns as the first row; append it to the list
		row_dict_list.append(row)
		row = None
	
	return row_dict_list, row

def get_log_information(path_to_log) :
	regex_iteration = re.compile('Iteration (\d+)')
	regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
	regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
	regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')
	iteration = -1
	learning_rate = float('NaN')
	train_dict_list = []
	test_dict_list = []
	train_row = None
	test_row = None
	
	with open(path_to_log) as f:
		for line in f:
			iteration_match = regex_iteration.search(line)
			if iteration_match:
				iteration = float(iteration_match.group(1))
			if iteration == -1:
				continue
			learning_rate_match = regex_learning_rate.search(line)
			if learning_rate_match:
				learning_rate = float(learning_rate_match.group(1))

			train_dict_list, train_row = parse_line_for_net_output(
    			regex_train_output, train_row, train_dict_list,
    			line, iteration, learning_rate)
			
			test_dict_list, test_row = parse_line_for_net_output(
    			regex_test_output, test_row, test_dict_list,
    			line, iteration, learning_rate)

	return train_dict_list, test_dict_list

def get_attention_plot_information(train_dict, test_dict, num_class) :
	"""
	Convert log information to meaningful information for plotting
	train_plot = [ iter, dir_loss, cls_loss ... ]
	test_plot  = [ iter, dir_acc, cls_acc, dir_loss, cls_loss ... ]
	"""
	train_iter = []
	train_dir_loss = []
	train_cls_loss = []
	for i in range(len(train_dict)) :
		train_iter.append( train_dict[i]['NumIters'] )
		dict_key = train_dict[i].keys()
		dir_loss = 0
		for k in xrange(2, len(dict_key)-1) :
			dir_loss += train_dict[i][dict_key[k]]
		train_dir_loss.append( dir_loss*0.33 )
		train_cls_loss.append( train_dict[i][dict_key[k+1]]*0.33 )
	train_plot = {}
	train_plot['iter'] = train_iter
	train_plot['dir_loss'] = train_dir_loss
	train_plot['cls_loss'] = train_cls_loss

	test_iter = []
	test_dir_acc = []
	test_cls_acc = []
	test_dir_loss = []
	test_cls_loss = []
	for i in range(len(test_dict)) :
		test_iter.append( test_dict[i]['NumIters'] )
		dict_key = test_dict[i].keys()
		# acc computation
		acc = 0
		for k in xrange(2, num_class*2+2) :
			acc += test_dict[i][dict_key[k]]
		test_dir_acc.append( acc/(2*num_class) )
		test_cls_acc.append( test_dict[i][dict_key[k+1]] )
		# loss computation
		dir_loss = 0
		for r in xrange(k+2, len(dict_key)-1) :
			dir_loss += test_dict[i][dict_key[r]]
		test_dir_loss.append( (dir_loss*0.33)/(num_class) )
		test_cls_loss.append( test_dict[i][dict_key[r+1]]*0.33 )
	test_plot = {}
	test_plot['iter'] = test_iter
	test_plot['dir_acc'] = test_dir_acc
	test_plot['cls_acc'] = test_cls_acc
	test_plot['dir_loss'] = test_dir_loss
	test_plot['cls_loss'] = test_cls_loss

	return train_plot, test_plot

def plot_attention_loss(train_plot, test_plot) :
	
	testing_freq = (test_plot['iter'][1]-test_plot['iter'][0]) / (train_plot['iter'][1]-train_plot['iter'][0])
	loss_y = train_plot['iter']
	
	pl.figure(1)
	test_loss = test_plot['dir_loss']
	train_loss = train_plot['dir_loss']
	test_loss = numpy.row_stack(test_loss)
	test_loss = numpy.tile(test_loss, (1, testing_freq))
	test_loss = list(test_loss.flatten())
	test_loss += [test_loss[-1]] * max(0,len(train_loss) - len(test_loss))
	test_loss = test_loss[:len(train_loss)]
	pl.plot(loss_y, train_loss, 'k-', label='Train', linewidth=0.75, marker='x')
	pl.plot(loss_y, test_loss,  'r-', label='Test' , linewidth=0.75)
	pl.legend(loc='best')
	pl.xlabel('Iter')
	pl.title('Direction Loss')

	pl.figure(2)
	test_loss = test_plot['cls_loss']
	train_loss = train_plot['cls_loss']
	test_loss = numpy.row_stack(test_loss)
	test_loss = numpy.tile(test_loss, (1, testing_freq))
	test_loss = list(test_loss.flatten())
	test_loss += [test_loss[-1]] * max(0,len(train_loss) - len(test_loss))
	test_loss = test_loss[:len(train_loss)]
	pl.plot(loss_y, train_loss, 'k-', label='Train', linewidth=0.75, marker='x')
	pl.plot(loss_y, test_loss,  'r-', label='Test' , linewidth=0.75)
	pl.legend(loc='best')
	pl.xlabel('Iter')
	pl.title('Classification Loss')
	pl.show()

def plot_attention_acc(test_plot) :
	
	acc_y = test_plot['iter']
	
	pl.figure(1)
	dir_acc = test_plot['dir_acc']
	cls_acc = test_plot['cls_acc']
	dir_acc = numpy.row_stack(dir_acc)
	dir_acc = list(dir_acc.flatten())
	cls_acc = numpy.row_stack(cls_acc)
	cls_acc = list(cls_acc.flatten())
	pl.plot(acc_y, dir_acc, 'k-', label='DIR', linewidth=1.0, marker='.')
	pl.plot(acc_y, cls_acc, 'r-', label='CLS', linewidth=1.0, marker='.')
	pl.legend(loc='best')
	pl.xlabel('Iter')
	pl.title('Accuracy (%)')
	pl.show()

if __name__ == '__main__':
	if len(sys.argv) != 2 :
		print "training log file path.."
		exit(1)
	
	path_to_log = sys.argv[1] #"bvlc_googlenet.log"
	train, test = get_log_information(path_to_log)
	train, test = get_attention_plot_information(train, test, 20)
	#plot_attention_loss(train, test)
	plot_attention_acc(test)


