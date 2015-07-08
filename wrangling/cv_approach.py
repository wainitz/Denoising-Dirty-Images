import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from copy import copy,deepcopy

WHITE = 255
BLACK = 0

def load_images(directory):
	filenames = glob.glob(directory + "*.png")
	images = { file_id(filename, directory) : cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in filenames}
	return images

def apply_transformation(image):
	return third_pass_filter(image)

def last_filter(img):
	x_index,y_index = img.shape
	for x in range(x_index):
		for y in range(y_index):
			if img[x,y]>60:
				img[x,y] = WHITE
			else:
				img[x,y] = BLACK

def third_pass_filter(image):
	clean = first_pass_mean_filter(image, num_stds=0.5)
	# now I need to remove coffee stains.
	removed_stains = sliding_window_stains(clean, 3, 30)
	img = filter_by_pixels(clean, removed_stains)
	img2 = cv2.erode(img,kernel=None)
	cv2.imshow('erode',img2)
	last_filter(img2)
	return filter_by_pixels(img, img2)

def second_pass_filter(image):
	'''
	score on training set = 0.072787758317669635
	gets me on ~20th place
	'''
	clean = first_pass_mean_filter(image, num_stds=0.5)
	# now I need to remove coffee stains.
	removed_stains = sliding_window_stains(clean, 7, 90)
	return filter_by_pixels(clean, removed_stains)
	

def sliding_window_stains(clean, width, window):
	(x_len, y_len) = clean.shape
	side_length = (width - 1)/2
	_, binary = cv2.threshold(clean, 0, 0, cv2.THRESH_BINARY) # massive hack, I just need a black image.
	for x in range(width, x_len - width):
		for y in range(width, y_len - width):
			middle_pixels = [clean[x+i][y+j] for i in range(-side_length, side_length+1) for j in range(-side_length, side_length+1)]
			max_color = max(middle_pixels)
			min_color = min(middle_pixels)
			if max_color - min_color <= window:
				binary[x][y] = 255
			else:
				binary[x][y] = 0
	return binary

def first_pass_mean_filter(image, num_stds=0.5):
	'''
	score on training set = 0.08277455124051479
	gets me on ~22th place
	'''
	thresh = threshold_by_mean(image, num_stds=num_stds)
	return filter_by_pixels(image, thresh)

def filter_by_pixels(image, threshold):
	'''
		Only keeps the black pixels from the image in threshold.
	'''
	binary = 1 - threshold / 255
	return threshold + np.multiply(image, binary)

def threshold_by_mean(image, num_stds=0):
	'''
		Filter by average color.
		seems to work with preserving letters. problem is that in heavy crumble, coffee stains, the image preserves. 
		it removes some creasing, some light darkness, some light shadow.
	'''
	mean_color = np.mean(image)
	std = np.std(image)
	_, threshold = cv2.threshold(image, mean_color - num_stds*std, 255, cv2.THRESH_BINARY)
	return threshold

def file_id(filename, directory):
	id_and_png = filename[len(directory):]
	png_index = id_and_png[:-4]
	return int(png_index)
	
def explore_transformation(dirty_dir, clean_dir, num_stds=0):
	images = load_images(dirty_dir)
	cleaned_images = load_images(clean_dir)
	for key in images.keys():
		image = images[key]
		image_c = third_pass_filter(image)
		cv2.imshow('original', image)
		cv2.moveWindow('original', 0, 0)
		cv2.imshow('cleaned by me', image_c)
		cv2.moveWindow('cleaned by me', 500, 0)
		them = cleaned_images[key]
		cv2.imshow('clean', them)
		cv2.moveWindow('clean', 500, 300)

		consider_white = 200
		important_me = np.extract(image_c.flatten() < consider_white, image_c.flatten())
		important_them = np.extract(them.flatten() < consider_white, them.flatten())

		bins = 20
		plt.hist(important_them, bins, label='them')
		plt.hist(important_me, bins, label='me')
		plt.legend(loc='upper right')
		#plt.show()

		cv2.waitKey(0)
		plt.close()
		cv2.destroyAllWindows()