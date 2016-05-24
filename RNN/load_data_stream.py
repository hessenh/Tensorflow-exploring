import pandas as pd
import numpy as np



class Data(object):
	"""docstring for Data"""
	def __init__(self, train_x, train_y, test_x, test_y):
		self._train_x = train_x
		self._train_y = train_y
		self._test_x = test_x
		self._test_y = test_y

		self._num_examples = len(train_y)
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def next_training_batch(self, batch_size, seq_length):
	    start = self._index_in_epoch
	    self._index_in_epoch += batch_size*seq_length
	    if self._index_in_epoch > self._num_examples:
	      # Finished epoch
	      self._epochs_completed += 1
	     
	      start = 0
	      self._index_in_epoch = batch_size*seq_length
	      assert batch_size <= self._num_examples
	    end = self._index_in_epoch
	    return self._train_x[start:end], self._train_y[start:end]




def main():
	print "Loading data"
	TRAIN_SUBJECTS = ['01A']#, '02A', '04A', '20A', '06A', '08A', '09A', '11A', '12A', '13A', '15A', '16A', '19A', '23A']
	TEST_SUBJECTS = ['21A']#, '05A', '14A', '18A', '03A', '22A', '10A']

	root_directory = '../../data/'
	subject = '01A'
	folder = 'RAW_SIGNALS'
	files = ['_Axivity_BACK_Back.csv', '_Axivity_THIGH_Right.csv', '_GoPro_LAB_All.csv']

	train_x, train_y = load_subject_data(TRAIN_SUBJECTS, root_directory, folder, files)
	test_x, test_y = load_subject_data(TEST_SUBJECTS, root_directory, folder, files)
	train_x, train_y = remove_activities(train_x, train_y)
	test_x, test_y = remove_activities(test_x, test_y)
	train_y = convert_y(train_y)
	test_y = convert_y(test_y)
	return Data(train_x, train_y, test_x, test_y)
	
def remove_activities(x, y):

	REMOVE_ACTIVITIES =  [0,3,9,11,16,12,15,17]

	new_x = []
	new_y = []
	keep_boolean = np.zeros(len(y), dtype=bool)
	for i in range(0, len(y)):
		if y[i] not in REMOVE_ACTIVITIES:
			keep_boolean[i] = True
	
	x = x[keep_boolean]
	y = y[keep_boolean]

	return x, y

def convert_y(y):
    #n = np.zeros(num_activities)
	CONVERTION = {1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:8, 13:9, 14:10}

	for i in range(0, len(y)):
		y[i] =  CONVERTION[y[i][0]]
	return y

def load_subject_data(SUBJECT_LIST, root_directory, folder, files):
	first_iteration = True
	# Iterate over all subjects
	for SUBJECT in SUBJECT_LIST:
		print SUBJECT
		path = root_directory + SUBJECT+ '/' + folder + '/' + SUBJECT

		df_0 = pd.read_csv(path  + files[0], header=None,engine='python')
		df_1 = pd.read_csv(path  + files[1], header=None,engine='python')
		
		x_temp = pd.concat([df_0, df_1],axis=1)
		y_temp = pd.read_csv(path + files[2], header=None, sep=',')

		length = len(y_temp)
		x_temp = x_temp[0:length]
		y_temp = y_temp[0:length]
		if first_iteration:
			x = x_temp.as_matrix()
			y = y_temp.as_matrix()
			first_iteration = False

		
		else:
			x = np.concatenate((x, x_temp), axis=0)
			y = np.concatenate((y, y_temp), axis=0)
	return x, y

if __name__ == "__main__":
    main()

