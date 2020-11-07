def FindMinMax(data):
	Mindata = np.min(data, 0)
	Maxdata = np.max(data, 0)
	
	return Mindata, Maxdata

def MinMaxScaler(data):
	numerator = data - np.min(data, 0)
	denominator = np.max(data, 0) - np.min(data, 0)
	
	return numerator / (denominator + 1e-7)

def MinMaxReturn(val, Min, Max):
	
	return val * (Max - Min + 1e-7) + Min
