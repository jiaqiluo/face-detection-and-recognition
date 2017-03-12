# Code Source:
# http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#appendixft

import sys
import os.path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":

  if len(sys.argv) != 3:
    print "usage: create_csv <base_path> <filename.csv>"
    sys.exit(1)
  if (".csv" not in sys.argv[2]):
      print "error: output needs to be saved as .csv file"
      print "Retry: create_csv <base_path> <filename.csv>"
      sys.exit(1)

  BASE_PATH = sys.argv[1]
  SEPARATOR = ";"

  label = 0
  temp = ""
  for dirname, dirnames, filenames in os.walk(BASE_PATH):
    for subdirname in dirnames:
      subject_path = os.path.join(dirname, subdirname)
      for filename in os.listdir(subject_path):
        abs_path = "%s/%s" % (subject_path, filename)
        temp +=  "%s%s%d\n" % (abs_path, SEPARATOR, label)
        # print "%s%s%d" % (abs_path, SEPARATOR, label)
      label = label + 1
  f = open(sys.argv[2], 'w')
  f.write(temp)
  f.close()
