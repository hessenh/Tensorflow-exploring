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

	def next_training_batch(self, batch_size):
	    start = self._index_in_epoch
	    self._index_in_epoch += batch_size
	    if self._index_in_epoch > self._num_examples:
	      # Finished epoch
	      self._epochs_completed += 1
	      # Shuffle the data
	      perm = np.arange(self._num_examples)
	      np.random.shuffle(perm)
	      self._train_x = self._train_x[perm]
	      self._train_y = self._train_y[perm]
	      # Start next epoch
	      start = 0
	      self._index_in_epoch = batch_size
	      assert batch_size <= self._num_examples
	    end = self._index_in_epoch
	    return self._train_x[start:end], self._train_y[start:end]




def main():
	print "Loading data"
	TRAIN_SUBJECTS = ['01A', '02A', '04A', '20A', '06A', '08A', '09A', '11A', '12A', '13A', '15A', '16A', '19A', '23A']
	TEST_SUBJECTS = ['21A', '05A', '14A', '18A', '03A', '22A', '10A']

	root_directory = '../../data/'
	subject = '01A'
	folder = 'DATA_WINDOW'
	window = '1.0'
	files = ['Axivity_BACK_Back_X.csv', 'Axivity_BACK_Back_Y.csv', 'Axivity_BACK_Back_Z.csv', 'Axivity_THIGH_Right_X.csv', 'Axivity_THIGH_Right_Y.csv', 'Axivity_THIGH_Right_Z.csv']
	
	train_x, train_y = load_subject_data(TRAIN_SUBJECTS, root_directory, folder, window)
	test_x, test_y = load_subject_data(TEST_SUBJECTS, root_directory, folder, window)

	train_x, train_y = remove_activities(train_x, train_y)
	test_x, test_y = remove_activities(test_x, test_y)

	train_y = convert_y(train_y)
	test_y = convert_y(test_y)

	return Data(train_x, train_y, test_x, test_y)
	
def remove_activities(x, y):
	
	REMOVE_ACTIVITIES =  [0,3,9,11,16,12,15,17]

	new_x = []
	new_y = []
	for i in range(0, len(y)):
		if y[i] not in REMOVE_ACTIVITIES:
			new_y.append(y[i])
			new_x.append(x[i])

	#for activity in REMOVE_ACTIVITIES:
	#	keep_boolean = y != activity
		
	#	x = x[keep_boolean]
	#	y = y[keep_boolean]
		#print sum(keep_boolean)
		#y = np.delete(y, keep_boolean, axis = 1)
		#x = np.delete(x, keep_boolean, axis = 1)
	return np.array(new_x), np.array(new_y)

def convert_y(y):
	num_activities = 10
	def convert(l):
	    n = np.zeros(num_activities)
	    CONVERTION = {1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:8, 13:9, 14:10}
	    if l in CONVERTION:
	        activity = CONVERTION[l]
	        n[activity-1] = 1.0
	    else:
	       	print "Not in convertion"
	    return n

    
	new_y = np.zeros([len(y),num_activities])
	for i in range(0, len(y)):
		new_y[i] =  convert(y[i][0])
	return new_y

def load_subject_data(SUBJECT_LIST, root_directory, folder, window):

	files = ['Axivity_BACK_Back_X.csv', 'Axivity_BACK_Back_Y.csv', 'Axivity_BACK_Back_Z.csv', 'Axivity_THIGH_Right_X.csv', 'Axivity_THIGH_Right_Y.csv', 'Axivity_THIGH_Right_Z.csv']
	first_iteration = True
	# Iterate over all subjects
	for SUBJECT in SUBJECT_LIST:
		print SUBJECT
		path = root_directory + SUBJECT+ '/' + folder + '/' + window + '/ORIGINAL'

		df_0 = pd.read_csv(path + '/' + files[0], header=None,engine='python')
		df_1 = pd.read_csv(path + '/' + files[1], header=None,engine='python')
		df_2 = pd.read_csv(path + '/' + files[2], header=None,engine='python')
		df_3 = pd.read_csv(path + '/' + files[3], header=None,engine='python')
		df_4 = pd.read_csv(path + '/' + files[4], header=None,engine='python')
		df_5 = pd.read_csv(path + '/' + files[5], header=None,engine='python')
		x_temp = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5],axis=1)
		
		y_temp = pd.read_csv(path + '/GoPro_LAB_All_L.csv', header=None, sep=',')

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

