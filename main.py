from Detector import *
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../images/image1.png"
abs_file_path = os.path.join(script_dir, rel_path)

detector = Detector()

detector.onImage("./input.jpg")