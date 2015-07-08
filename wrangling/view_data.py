import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from cv_approach import *

DIRTY_DIR = 'data/training_set/'
CLEAN_DIR = 'data/train_cleaned/'

def loop_through_images_in_dir(dir_name):
	files = glob.glob(dir_name + "*.png")
	for filename in files:
		img = mpimg.imread( filename )
		plt.imshow(img)

def view_images_and_histogram(dirty_dir, clean_dir):
	dirty = load_images(dirty_dir)
	clean = load_images(clean_dir)
	for key in dirty.keys():
		d_image = dirty[key]
		c_image = clean[key]

		bins = 20
		plt.hist(d_image.flatten(), bins, label='d_image')
		plt.figure()
		plt.hist(c_image.flatten(), bins, label='c_image')
		plt.figure()
		plt.show()


def test_one():
	dirty = load_images(DIRTY_DIR)
	image = dirty[2]
	clean = apply_transformation(image)
	cv2.imshow('original',image)
	cv2.imshow('cleaned_by_wainitz',clean)
	cv2.imshow('cleaned_by_them',load_images(CLEAN_DIR)[2])