import featureextraction as fe
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import shutil
import numpy as np

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


  def pickle_numpy_matrix_for_features(self, input_path, pickled_file):
   
    file_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    tempdict = defaultdict(float)
    global_feats_list=[]
    i=0

    for file in file_names:
      if '_' not in file:
        continue
      if "DS_Store" in file:
      	continue
      if file.startswith("."):
        continue

      image_file = join(input_path, file)
      imhist = fe.get_color_histogram_from_image(image_file)
      descriptors = fe.get_GIST_from_image(image_file)
      global_feat = np.concatenate((imhist, descriptors))
      global_feats_list.append(global_feat)
      print "Doing for image " + str(i)
      i+=1

    global_feat_matrix = np.array(global_feats_list)
    np.save(pickled_file, global_feat_matrix)
    print global_feat_matrix.shape





if __name__ == '__main__':
    fe_generator = FeatureGenerator()
    #f1 = file("train.bin","wb")
    f2 = file("val.bin","wb")
    f3 = file("test.bin","wb")
    #fe_generator.pickle_numpy_matrix_for_features("/Users/Ankush/Desktop/DeepLearning/Project/data/PGH/PGH_Train", f1)
    fe_generator.pickle_numpy_matrix_for_features("/Users/Ankush/Desktop/DeepLearning/Project/data/PGH/PGH_Val", f2)
    fe_generator.pickle_numpy_matrix_for_features("/Users/Ankush/Desktop/DeepLearning/Project/data/PGH/PGH_Test", f3)
    #f1.close()
    f2.close()
    f3.close()
   

    """
    for i in xrange(8,11):
      input_path = "/Users/Ankush/Desktop/part" + str(i)
      data_gatherer.get_city_from_image_name(input_path)
    """



