
from geopy.geocoders import Nominatim
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import shutil

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
    return str(raw_address['address']['city'])



  def copy_file(self, input_path, file_name, output_path, city):

    if city is not None:
      dst_dir = join(output_path, city)
    else:
      dst_dir = output_path

    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)

    src_file = join(input_path, file_name)
    shutil.move(src_file, dst_dir)


  def get_address_from_image_name(self, input_path):
    coordinates_dict = self.get_coordinates_from_file()
    placemarker_to_address = {}
    city_counts = defaultdict(float)
    output_path = "/Users/Ankush/Desktop/cities2"
    failed_path = "/Users/Ankush/Desktop/failed_files3"

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
      if placemark_identifier in placemarker_to_city:
        address = placemarker_to_city[placemark_identifier]
        print "Got this address from memory -->" + city
      else:
        try:
          coordinates = coordinates_dict[placemark_identifier]
          latitude = coordinates["lat"]
          longtitude = coordinates["long"]
          city = self.get_city(latitude, longtitude)
          print "Got this city from api -->" + city
          placemarker_to_city[placemark_identifier] = city

        except:
          print "Failed reading city for file ->" + file + ". Copying it to different loc"
          self.copy_file(input_path, file, failed_path, None)
          continue

      city_counts[city]+=1
      self.copy_file(input_path, file, output_path, city)
      print city_counts


if __name__ == '__main__':
    data_gatherer = DataGatherer()
    data_gatherer.get_address_from_image_name("/Users/Ankush/Desktop/DeepLearning/Project/FineGrained/NYC_Complete")

    """
    for i in xrange(8,11):
      input_path = "/Users/Ankush/Desktop/part" + str(i)
      data_gatherer.get_city_from_image_name(input_path)
    """



