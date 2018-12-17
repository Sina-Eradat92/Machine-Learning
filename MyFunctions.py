from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil 
import os 


class MyFunctions:
	
	# Encode text values to dummy variables ie(red, green, blue -> [1,0,0], [0,1,0], [0,0,1])
	def encode_text_dummy(df, name):
		dummies = pd.get_dummies(df[name])
		for x in dummies.columns:
			dummy_name = f"{name}-{x}"
			df[dummy_name] = dummies[x]
		df.drop(name, axis=1, inplace=True)
	
	# Encode text values to a single dummy variable. The new columns (which do not replace the old) will have a 1
	# at every location where the original column (name) matches each of the target_values. One column is added for
	# each target value 
	def encode_text_single_dummy(df, name, target_values):
		for tv in target_values:
			l = list(df[name].astype(str))
			l = [l if str(x) == str(tv) else 0 for x in l]
			name2 = f"{name}-{tv}"
			df[name2] = 1
	
	# Encode text values to indexes ie:([1],[2],[3] -> red, green, blue)
	def encode_text_index(df, name):
		le = preprocessing.LabelEncoder()
		df[name] = le.fit_transform(df[name])
		return le.classes_
	
	# Encode a numeric column as Zscore
	def encode_numeric_zscore(df, name, mean=None, sd=None):
		if mean is None:
			mean = df[name].mean()
		
		if sd is None:
			sd = df[name].std()
		
		df[name] = (df[name] - mean) / sd
	
	# Convert all missing values in the column to the median
	def missing_median(df, name):
		med = df[name].median()
		df[name] = df[name].fillna(med)
	
	
	# Convert all missing values in the column to the default given
	def misssing_default(df, name, default_value):
		df[name] = df[name].fillna(default_value)
	
	# Convert a pandas dataframe to x,y input for tensorflow where target is y
	def to_xy(df, target):
		
		dummy_df = df.drop(target,axis=1)
		
		target_type = df[target].dtypes
		target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
		
		if target_type in(np.int64, np.int32):
			#Classification
			dummies = pd.get_dummies(df[target])
			return dummy_df.values.astype(np.float32), dummies.values.astype(np.float32)
		else:
			#Regression
			dummy = df.filter([target], axis=1)
			return dummy_df.values.astype(np.float32), dummy.values.astype(np.float32)
	
	#
	# seq_size = , obs = 
	def to_sequences(seq_size, obs):
		x = []
		y = []
		
		for i in range(len(obs)-seq_size-1):
			window = obs[i : (i + seq_size)] 
			after_window = obs[i + seq_size]
			window = [[x] for x in window]
			x.append(window)
			y.append(after_window)
			
		return np.array(x), np.array(y)
	
	# Formated Time String
	def hms_string(sec_elapsed):
		h = int(sec_elapsed/(60*60))
		m = int((sec_elapsed%(60*60))/60)
		s = sec_elapsed % 60
		return f"{h}:{m:>02}:{s:05.2f}"
	
	# Regression Chart
	def chart_regression(pred,y,sort=True):
		t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
		if sort:
			t.sort_values(by=['y'], inplace=True)
		a = plt.plot(t['y'].tolist(), label='expected')
		b = plt.plot(t['pred'].tolist(), label='prediction')
		plt.ylabel('output')
		plt.legend()
		plt.show()
	
	# Remove all rows where the selected column is +/- sd
	def remove_outliers(df, name, sd):
		drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
		df.drop(drop_rows, axis=0, inplace=True)
		
	# Encode a column to a range between normalized_low and normalized_high
	def encode_numeric_range(df, name, normalized_low=1, normalized_high=1, data_low=None, data_high=None):
		if data_low is None:
			data_low = min(df[name])
			data_high = max(df[name])
		
		df[name] = ((df[name] - data_low)/ (data_high - data_low)) \
		           * (normalized_high - normalized_low) + normalized_low
		
	# Plot a confusion matrix (for any classification).
	# cm is the confusion matrix, names are the names of the classes
	def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(names))
		plt.xticks(tick_marks, names, rotation=45)
		plt.yticks(tick_marks, names)
		plt.tight_layout()
		plt.ylabel('True Label')
		plt.xlabel('Predicted label')
		plt.show()

	#  Compute confusion matrix (pridiction, argmax(y))
	def  compute_confusion_matrix(pred, y_compare, normaliz=True):
		cm = confusion_matrix(y_compare, pred)
		if normaliz:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print('Normalized confusion matrix')
		else:
			np.set_printoptions(precision=2)
			print('Confusion matrix, without normalization')
		print(cm)
		return cm
	
	# Plot a ROC this is for regression. (pred - the prediction, y - the expected output)
	def plot_roc(pred, y):
		fpr, tpr, _ = roc_curve(y, pred)
		roc_auc = auc(fpr, tpr)
		
		plt.plot(fpr, tpr, label='area')
		plt.plot([0,1], [0,1], 'k--')
		plt.xlim([0.0,1.0])
		plt.ylim([0.0,1.5])
		plt.xlabel('False Positive Rate')
		plt.ylabel('The Positive Rate')
		plt.title('Receiver operating Characteristic (ROC)')
		plt.legend(loc='lower right')
		plt.show()
	
	# Display Images 
	def display_image(x_train, y_train, index=0, color='gray'):
		print(f"Image (#{index}): Which is '{y_train[index]}'")
		plt.imshow(x_train[index], cmap=color, interpolation='nearest')
		plt.show()
	
	# Evaluate and Display the coefficients of a regression
	def report_coef(names, coef, intercept):
		r = pd.DataFrame({'coef': coef, 'positive': coef>=0}, index = names)
		r = r.sort_values(by=['coef'])
		print(r)
		print(f"Intercept: {intercept}")
		r['coef'].plot(kind='barh', color=r['positive'].map({True: 'b', False: 'r'}))
		plt.show()
	#
	
	
	
	
	
	
	