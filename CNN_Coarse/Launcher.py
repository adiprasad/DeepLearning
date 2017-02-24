import numpy as np
import matplotlib.pyplot as plt
from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *
import asgn2.classifiers.cnn
from asgn2.classifiers.cnn import MyCustomConvNet
import matplotlib.pyplot as plt
from asgn2.solver import Solver
import traceback
import os
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
	X_train = np.load("Downsampled_Train_Data/X_train.npy")
	Y_train = np.load("Downsampled_Train_Data/Y_train.npy")
	X_val = np.load("Downsampled_Val_Data/X_val.npy")
	Y_val = np.load("Downsampled_Val_Data/Y_val.npy")
	print "X train shape : ", X_train.shape
	print "Y train shape : ", Y_train.shape
	print "X val shape : ", X_val.shape
	print "Y val shape : ", Y_val.shape

	data = {
	  'X_train': X_train,
	  'y_train': Y_train,
	  'X_val': X_val,
	  'y_val': Y_val,
	}

	
	plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'


	model = MyCustomConvNet(weight_scale=0.001, hidden_dim=100, reg=0.0001,use_saved_weights=True)

	# 400 batch size, epochs = 5, iterations = 33 x 5
	solver = Solver(model, data,
	                num_epochs=3, batch_size=200,
	                update_rule='adam',
	                optim_config={
	                  'learning_rate': 0.001134,
	                },
                    lr_decay=0.90,
	                verbose=True, print_every=10)

	try:
	    solver.train()
	except:
	    traceback.print_exc()         
 

	
    # Save parameters of each layer of network
	for key,val in solver.model.params.items():
		f_path = join("Write_Params" , key + ".npy")
		fl = open(f_path,"w")
		np.save(fl,val)
		fl.close()

	# Save bn_parameters:

	for i in xrange(len(solver.model.bn_params)):
          bn_param = solver.model.bn_params[i]
          running_mean = bn_param['running_mean'] 
          running_var = bn_param['running_var']

          f_path = join("Write_Params" , 'running_mean_' + str(i)  + ".npy")
          fl = open(f_path,"w")
          np.save(fl,running_mean)
          fl.close()
          f_path = join("Write_Params" , 'running_var_' + str(i)  + ".npy")
          fl = open(f_path,"w")
          np.save(fl,running_var)
          fl.close()

	# Also save loss_history, train_ac_history, test_ac_history
	f_path1 = join("Write_Params" , "loss_history.npy")
	f_path2 = join("Write_Params" , "train_ac_history.npy")
	f_path3 = join("Write_Params" , "val_ac_history.npy")

	fl1 = open(f_path1,"w")
	fl2 = open(f_path2,"w")
	fl3 = open(f_path3,"w")

	np.save(fl1,solver.loss_history)
	np.save(fl2,solver.train_acc_history)
	np.save(fl3,solver.val_acc_history)

	fl1.close()
	fl2.close()
	fl3.close()




