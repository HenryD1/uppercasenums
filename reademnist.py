from emnist import list_datasets
from emnist import extract_training_samples
import numpy as np

images, labels = extract_training_samples('byclass')

#for index, image in enumerate(images, start=0):
	#arr = np.append(image, np.array([[1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]), axis = 0)
	#print(index, 'begin', arr, 'end')

def get_letter(index: int) -> int:
	label = labels[index] #[] is because labels is a list.
	print(label)
	if ((label < 11) or (label > 36)):
		print('you have attempted to get the letter of a training image which is not a lowercase letter')
	return (label)

def get_uppercase(index: int) -> int:
	label = labels[index]
	if ((label < 11) or (label > 36)):
		print('you have attempted to get the uppercase counterpart of a training image which is not a lowercase letter')
	return (index + 26)
	#its probably 1-10 + 26*2
	#yeah so if its in [10,36] just add 26, else say not a lowercaseletter

def append(lower, upper):
	for row in upper:
		row = np.reshape(row, (-1, 28))
		print(row)
		lower = np.append(lower, row, axis = 0)
	print('commence new lower')
	for row in lower:
		print(row)

def preprocesslabels(labels):
	letterindices ={}
	for index, letter in enumerate(labels, start = 0):
		if (letter in letterindices):
			#print(letter, 'heres the problem officer', letterindices[letter])
			letterindices[letter].append(index) #lol no idea if this works
			#print('is this none?', letterindices[letter])
		else:
			letterindices[letter] = []
			letterindices[letter].append(index)
	return letterindices
	#arranges labels into dict with all indices at which a letter is found stored at the key that corresponds to the letter


def get_all_lowercase_letters():
	#returns a n x 28 x 28 file ready to be input of generative network, each square a lowercase real letter
	lowercaseletters = []
	letterindices = preprocesslabels(labels)
	for letter in [10, 36]:
		for index in letterindices[letter]:
			lowercaseletters.append(images[index])
			print(lowercaseletters)
	print(lowercaseetters)

get_all_lowercase_letters() #hmm I'm not sure if this did exactly what I wanted it to;



#it lists the datatype which is awesome but idk if that's supposed to be there. Hopefully tensorflow will be chill with it. 