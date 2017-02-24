
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import shutil
import numpy as np
import pickle

class FeatureGenerator:

  def __init__(self):
    pass


  def copy_file(self, input_path, file_name, output_path, city):

    if city is not None:
      dst_dir = join(output_path, city)
    else:
      dst_dir = output_path

    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)

    src_file = join(input_path, file_name)
    shutil.move(src_file, dst_dir)


  def print_unique_key(self, input_path, key ):
   
    file_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    tempdict = defaultdict(float)

    global_feats_list=[]
    i=0

    unique_places = set()

    for file in file_names:
      if '_' not in file:
        continue
      if "DS_Store" in file:
      	continue
      if file.startswith("."):
        continue

      placemarker = file[:7]
      if placemarker in unique_places:
        continue

      unique_places.add(placemarker)
      src_file_name = join(input_path, file)
      file_name = open(src_file_name, 'rb')
      address_dict = pickle.load(file_name)

      if key not in address_dict:
        continue

      value = address_dict[key]
      tempdict[value] += 1


    for key in tempdict:
      print key + "  " + str(tempdict[key])

    print len(tempdict)

if __name__ == '__main__':
    fe_generator = FeatureGenerator()
    fe_generator.print_unique_key("/Users/Ankush/Desktop/DeepLearning/Project/NYC_Address", 'suburb')


