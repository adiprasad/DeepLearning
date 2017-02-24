import numpy as np
import scipy.misc
from scipy.misc import imread
import random
import os 
from os import listdir
from os.path import isfile, join

class Train_Val_Maker(object) :

	def __init__(self):
		self.file_cnt = 0

	def city_preparator(self,input_path,class_label):
		img_list = []
		
		file_names = [f for f in listdir(input_path) if isfile(join(input_path, f)) and ("DS_Store" not in f)]
		ylabel = input_path.split("/")[-1]
		label_file_cnt = 0

		for file in file_names:
			dest_path = join(input_path, file)
			img = imread(dest_path)
			img_list.append(img)

			label_file_cnt += 1
			self.file_cnt += 1

			print "%s file count : %d, Total file count : %d" %(ylabel, label_file_cnt, self.file_cnt)


		num_imgs = len(img_list)

		X = np.stack(img_list,axis=3)
		Y = np.full(num_imgs,class_label,dtype=np.int)
 
		X = X.swapaxes(0,3).swapaxes(1,2).swapaxes(2,3)		# Bringing it to the format accepted by CNN

		return X,Y


	def	randomize(self,X,Y):
		rand_sequence = random.sample(range(0,X.shape[0]),X.shape[0])
		randomized_X = X[rand_sequence,]			# Randomize X
		randomized_Y = Y[rand_sequence,]			# Randomize Y 

		return randomized_X, randomized_Y


	def data_aggregator(self,X_array,Y_array):			# Pass on arrays of city Xs and city Ys to aggregate 
		X_stacked = np.vstack(X_array)
		Y_stacked = np.hstack(Y_array)

		self.X, self.Y = self.randomize(X_stacked,Y_stacked)


	def dump_datasets(self,output_path,mode):

		if not os.path.exists(output_path):
			os.makedirs(output_path)

		X_path = join(output_path, "X_"+mode+".bin")
		Y_path = join(output_path, "Y_"+mode+".bin")

		print "X before dumping : ",self.X.shape
		print "Y before dumping : ",self.Y.shape

		np.save(X_path, self.X)
		np.save(Y_path, self.Y)

if __name__ == '__main__':
	preparator = Train_Val_Maker()

	NYC_Train_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/NYC/NYC_Downsampled/NYC_Val"
	Orlando_Train_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Orlando/Orlando_Downsampled/Orlando_Val"
	PGH_Train_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/PGH/PGH_Downsampled/PGH_Val"
	output_path = "/Users/Ankush/Desktop"
	
	NYC_X,NYC_Y = preparator.city_preparator(NYC_Train_path,2)
	Orlando_X,Orlando_Y = preparator.city_preparator(Orlando_Train_path,1)
	PGH_X,PGH_Y = preparator.city_preparator(PGH_Train_path,0)
	
	preparator.data_aggregator([NYC_X,Orlando_X,PGH_X],[NYC_Y,Orlando_Y,PGH_Y])
	
	preparator.dump_datasets(output_path,"val")
