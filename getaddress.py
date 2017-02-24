
from geopy.geocoders import Nominatim
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import shutil
import pickle

class DataGatherer:

  def __init__(self):
    self.geolocator = Nominatim()


  def get_coordinates_from_file(self):
    coordinates_dict = {}
    fname = '/Users/Ankush/Desktop/DeepLearning/Project/FineGrained/data.txt'
    with open(fname) as f:
      content_list = f.readlines()

    for index in xrange(len(content_list)):
      coordinates = content_list[index]
      #print coordinates
      coordinates_list = coordinates.split(',')
      coordinates_dict[index + 1] = {"lat":coordinates_list[0], "long":coordinates_list[1]}

    return coordinates_dict



  def get_address(self, latitude, longitude):
    s = str(latitude) + "," + str(longitude)
    location =  self.geolocator.reverse(s)
    raw_address = location.raw
    return (raw_address['address'])


  def create_address_file(self, output_path, file_name, address):

    if not os.path.exists(output_path):
      os.makedirs(output_path)

    file_name = file.split(".")[0] + ".p"
    pickle.dump(address, open(file_name, "wb"))


  def copy_failed_file(self, input_path, file_name, output_path):

    if not os.path.exists(output_path):
      os.makedirs(output_path)

    src_file = join(input_path, file_name)
    shutil.copy(src_file, output_path)


  def get_address_from_image_name(self, input_path):
    coordinates_dict = self.get_coordinates_from_file()
    output_path = "/Users/Ankush/Desktop/DeepLearning/Project/FineGrained/NYC_Address"
    failed_path = "/Users/Ankush/Desktop/DeepLearning/Project/FineGrained/Failed_Files"
    placemarker_to_address = {}
    file_count = 0

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
      if "_5" in file:
        continue

      placemark_identifier = int(file.split("_")[0])
      file_count +=1
      print "Doing for file " + str(file_count)
      if placemark_identifier in placemarker_to_address:
        address = placemarker_to_address[placemark_identifier]
        print "Got the address from memory for file-->" + file
      else:
        try:
          coordinates = coordinates_dict[placemark_identifier]
          latitude = coordinates["lat"]
          longtitude = coordinates["long"]
          address = self.get_address(latitude, longtitude)
          print "Got the address from api -->" + file
          placemarker_to_address[placemark_identifier] = address

        except:
          print "Failed reading address for file ->" + file + ". Copying it to different loc"
          self.copy_failed_file(input_path, file, failed_path)
          continue

      self.create_address_file(output_path, file, address)

if __name__ == '__main__':
    data_gatherer = DataGatherer()
    data_gatherer.get_address_from_image_name("/Users/Ankush/Desktop/DeepLearning/Project/FineGrained/NYC_Complete")


