

def append(lower, upper):  #lmao if you just dont declare the type then thats fine somehow
	if (len(lower[1]) != len(upper[1])):
		print('error: arrays have different dimensions')
		return 0
	for index, row in enumerate(upper, start = 0):
		print(row)
		print(upper[index])
		lower = np.append(lower, row, axis = 0)
	return lower
