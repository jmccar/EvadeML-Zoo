import urllib.request
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(CURRENT_DIR)

filename = ['test_32x32.mat','train_32x32.mat']
addr = ['http://ufldl.stanford.edu/housenumbers/test_32x32.mat','http://ufldl.stanford.edu/housenumbers/train_32x32.mat']

for i in range(2):
	f = os.path.join(CURRENT_DIR,filename[i])
	if not os.path.exists(f):
		output = open(f,'w')
		output.write(urllib.request.urlopen(addr[i]).read())
		output.close()
