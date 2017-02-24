from PIL import Image
import numpy
import os
from os import listdir
from os.path import isfile, join
import shutil
class DownSampler:

  def __init__(self):
    self.height = 227
    self.width = 227


  def down_sample(self, input_path, output_path):
    file_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    i = 0
    for file in file_names:
      if '_' not in file:
        continue
      if "DS_Store" in file:
        continue
      if file.startswith("."):
        continue
      
      i+=1
      print "Doing for image " + str(i)
      image_file = join(input_path, file)
      img = Image.open(image_file)
      out_img = img.resize( (self.height, self.width), Image.ANTIALIAS)
      out_img_path = join(output_path, file)
      out_img.save(out_img_path)


if __name__ == '__main__':
    ds = DownSampler()
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/PGH/PGH_Train", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/PGH/PGH_Train")
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/PGH/PGH_Val", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/PGH/PGH_Val")
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/PGH/PGH_Test", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/PGH/PGH_Test")

    ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Orlando/Orlando_Train", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/Orlando/Orlando_Train")
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Orlando/Orlando_Val", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/Orlando/Orlando_Val")
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Orlando/Orlando_Test", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/Orlando/Orlando_Test")

    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/NYC/NYC_Train", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/NYC/NYC_Train")
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/NYC/NYC_Val", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/NYC/NYC_Val")
    #ds.down_sample("/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/NYC/NYC_Test", "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Downsampled/NYC/NYC_Test")




