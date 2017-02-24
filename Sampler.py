
from geopy.geocoders import Nominatim
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import shutil
import random

class Sampler:

  def __init__(self):
    pass


  def preprocess_files(self, input_dir):
    processed_files_names = []
    file_names = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    for file in file_names:
      if '_' not in file:
        continue
      if "DS_Store" in file:
        continue
      if file.startswith("."):
        continue 
      if "_0" in file:
        continue
      if "_5" in file:
        continue

      processed_files_names.append(file)
    return processed_files_names


  def sample_and_copy_files(self, input_dir, output_dir):
    processed_file_names = self.preprocess_files(input_dir)
    random_files = random.sample(processed_file_names, 800)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for file in random_files:
      src_file = join(input_dir, file)
      shutil.move(src_file, output_dir)


if __name__ == '__main__':
    sampler = Sampler()
    sampler.sample_and_copy_files("/Users/Ankush/Desktop/DeepLearning/Project/PGH_Sampled", "/Users/Ankush/Desktop/DeepLearning/Project/PGH_Val")

    """
    for i in xrange(8,11):
      input_path = "/Users/Ankush/Desktop/part" + str(i)
      data_gatherer.get_city_from_image_name(input_path)
    """



