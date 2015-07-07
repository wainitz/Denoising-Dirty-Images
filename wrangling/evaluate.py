from sklearn.metrics import mean_squared_error
from cv_approach import *
import numpy as np

TRAINING_DIRTY_DIR = 'data/training_set/'
TRAINING_CLEAN_DIR = 'data/train_cleaned/'

def compute_rmse(clean, dirty):
	scaled_clean = clean / 255
	scaled_dirty = dirty / 255
	return mean_squared_error(scaled_clean, scaled_dirty)**0.5

def run_predictions_and_score(dirty_pics_dir, cleaned_pics_dir):
	images = clean_images(dirty_pics_dir)
	score = calculate_score(images, cleaned_pics_dir)
	return score

def clean_images(directory):
	images = load_images(directory)
	print "cleaning images..."
	cleaned_images = { key : apply_transformation(images[key]) for key in images.keys() }
	return cleaned_images

def calculate_score(images, cleaned_pics_dir):
	print "loading cleaned images to compute score..."
	cleaned = load_images(cleaned_pics_dir)
	testing_stack = np.array([])
	cleaned_stack = np.array([])
	print "getting data ready..."
	for key in images.keys():
		testing_stack = np.hstack( (testing_stack, images[key].flatten()))
		cleaned_stack = np.hstack( (cleaned_stack, cleaned[key].flatten()))
	print "computing score...."
	return compute_rmse(cleaned_stack, testing_stack)


