
from geopy.geocoders import Nominatim
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import shutil
import numpy as np
from numpy import genfromtxt
import re

class color_hist_getter:

  def __init__(self):
    self.geolocator = Nominatim()


  def get_colorhist_from_file(self):
    coordinates_dict = {}
    fname = '/Users/Ankush/Desktop/DeepLearning/Project/data/Sampled2/temp2.text'
    my_data = genfromtxt(fname, delimiter=',')
    return my_data


  def get_colorhist_from_image_name(self, input_path, pickled_file):
    colorhist_array = self.get_colorhist_from_file()
    colorhist_list = []

    file_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]

    for file in file_names:
      print file
      if '_' not in file:
        continue
      if "DS_Store" in file:
      	continue
      if file.startswith("."):
        continue
      if "_0" in file:
        continue

      temp = re.split("[_\.]", file)
      row_num = int(temp[0] + temp[1])
      colorhist_row = colorhist_array[row_num -1]
      if np.mean(colorhist_row) == 0:
        print "BOOOM BOOOOOOOOOOOOOOOOOOOOOOOOOOOOM----------------"
      colorhist_list.append(colorhist_row.tolist())

    global_feat_matrix = np.array(colorhist_list)
    np.save(pickled_file, global_feat_matrix)
    print global_feat_matrix.shape


if __name__ == '__main__':
    histgetter = color_hist_getter()
    f1 = file("train.bin","wb")
    f2 = file("val.bin","wb")
    f3 = file("test.bin","wb")
    histgetter.get_colorhist_from_image_name("/Users/Ankush/Desktop/DeepLearning/Project/data/Sampled2/NYC_Train", f1)
    histgetter.get_colorhist_from_image_name("/Users/Ankush/Desktop/DeepLearning/Project/data/Sampled2/NYC_Val", f2)
    histgetter.get_colorhist_from_image_name("/Users/Ankush/Desktop/DeepLearning/Project/data/Sampled2/NYC_Test", f3)
    f1.close()
    f2.close()
    f3.close()



