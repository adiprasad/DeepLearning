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
		self.img_name_list = []			# Introduced for test

	def city_preparator(self,input_path,class_label):
		img_list = []
		
		file_names = [f for f in listdir(input_path) if isfile(join(input_path, f)) and ("DS_Store" not in f)]
		ylabel = input_path.split("/")[-1]
		label_file_cnt = 0

		for file in file_names:
			dest_path = join(input_path, file)
			img = imread(dest_path)
			img_list.append(img)
			self.img_name_list.append(file)		# Introduced for test

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

		#self.X, self.Y = self.randomize(X_stacked,Y_stacked)
		self.X, self.Y = X_stacked,Y_stacked			# Introduced for test, to get back to normal, delete this and uncomment the line above


	def dump_datasets(self,output_path,mode):

		if not os.path.exists(output_path):
			os.makedirs(output_path)


		# X_path = join(output_path, "X_"+mode+".dat")
		# Y_path = join(output_path, "Y_"+mode+".dat")

		X_path = join(output_path, "X_"+mode+".npy")
		Y_path = join(output_path, "Y_"+mode+".npy")
		names_file_path = join(output_path, "Names_"+mode+".npy")			# Introduced for test

		file_x = open(X_path,"w")
		file_y = open(Y_path,"w")
		file_names = open(names_file_path,"w")				# Introduced for test

		print "X before dumping : ",self.X.shape
		print "Y before dumping : ",self.Y.shape
		print "Names file before dumping : ", len(self.img_name_list)			# Introduced for test

		# self.X.dump(X_path)
		# self.Y.dump(Y_path)

		np.save(file_x,self.X)
		np.save(file_y,self.Y)
		np.save(file_names,self.img_name_list)			# Introduced for test

		file_x.close()
		file_y.close()
		file_names.close()					# Introduced for test


if __name__ == '__main__':
	preparator = Train_Val_Maker()

	output_path = "/Users/Ankush/Desktop/numpy_data"

	class_0_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/0/Test"
	class_1_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/1/Test"
	class_2_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/2/Test"
	class_3_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/3/Test"
	class_4_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/4/Test"
	class_5_train_p = "//Users/Ankush/Desktop/Downsampled_Data_256_splits/5/Test"
	class_6_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/6/Test"
	class_7_train_p = "/Users/Ankush/Desktop/Downsampled_Data_256_splits/7/Test"
	
	class_0_X,class_0_Y = preparator.city_preparator(class_0_train_p,0)
	class_1_X,class_1_Y = preparator.city_preparator(class_1_train_p,1)
	class_2_X,class_2_Y = preparator.city_preparator(class_2_train_p,2)
	class_3_X,class_3_Y = preparator.city_preparator(class_3_train_p,3)
	class_4_X,class_4_Y = preparator.city_preparator(class_4_train_p,4)
	class_5_X,class_5_Y = preparator.city_preparator(class_5_train_p,5)
	class_6_X,class_6_Y = preparator.city_preparator(class_6_train_p,6)
	class_7_X,class_7_Y = preparator.city_preparator(class_7_train_p,7)
	

	preparator.data_aggregator([class_0_X,class_1_X,class_2_X,class_3_X,class_4_X,class_5_X,class_6_X,class_7_X],[class_0_Y,class_1_Y,class_2_Y,class_3_Y,class_4_Y,class_5_Y,class_6_Y,class_7_Y])
	
	preparator.dump_datasets(output_path,"test")





