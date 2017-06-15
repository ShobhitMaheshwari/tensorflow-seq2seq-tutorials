import collections, pandas
import nltk, os.path, pickle, random, numpy as np

def vocab(name=None):
	if(os.path.isfile('voc')):
		return pickle.load( open( "voc", "rb" ))
	#voc, rev_voc = vocab(name)
	voc = []
	o = pandas.read_json('pandas.txt')
	for index, row in o.iterrows():
		for sent in nltk.sent_tokenize(row['content']):
#	with open(name, 'r') as f:
#		for line in f:
#			for sent in nltk.sent_tokenize(line):
			for word in nltk.word_tokenize(sent):
				voc.append(word.lower())
	counter=collections.Counter(voc)

	counter2 = [x for x,y in counter.most_common(40000)]
	voc = {v:idx+3 for (idx, v) in enumerate(counter2)}
	rev_voc = {idx+3:v for (idx, v) in enumerate(counter2)}
	pickle.dump((voc, rev_voc), open("voc", "wb" ))
	return voc,rev_voc

def sent_token(voc, sent):
	s = []
	for word in nltk.word_tokenize(sent):
		if word.lower() in voc:
			s.append(voc[word.lower()])
		else:
			s.append(2)
	return s

def bucket(pair, B=[(5, 10), (10, 15), (15, 20), (20, 30)]):
	m = {}
	for idx, (a,b) in enumerate(B):
		m[idx]=[]
	for x,y,z in pair:
		for idx, (b,a) in enumerate(B):
			if(len(x)<a and len(y)<a and len(z)<b):
				m[idx].append((x,y,z))
				break
	return m

def most_unknowns(tokenvec):
	if(tokenvec.count(2) > .3*len(tokenvec)):
		return True
	return False

def test(voc, sent):
	w=sent_token(voc,sent)
	print(w)
	print(most_unknowns(w))

def readf(name=None):
	if(os.path.isfile('data')):
		return pickle.load( open( "data", "rb" ) )
	pair = []
	voc, iv = vocab(name)
	print('obtained vocab')
	o = pandas.read_json('pandas.txt')
	c=0
	k=0
	for index, row in list(o.iterrows()):
		sents = []
		for sent in nltk.sent_tokenize(row['content']):
			sents.append(sent)
		c+=len(sent)
		for i in range(0, len(sents[:-2])):
			x,y,z = sent_token(voc,sents[i]),sent_token(voc,sents[i+1]), sent_token(voc,sents[i+2])
			if(len(x)==0 or len(y)==0 or len(z)==0): 
				print(str(x) + "__" + str(y) + "__" + str(z))
				continue
			if(most_unknowns(x) or most_unknowns(y) or most_unknowns(z)): 
				print(sents[i] + "__" + sents[i+1] + "__" + sents[i+2])
				print(str(x) + "__" + str(y) + "__" + str(z))
				continue
			k+=1
			pair.append((x,y,z))
	print(str(c)+"_"+str(k))
	"""
	with open(name, 'r') as f:
		fff = f.read().split('\n')
	ff=[]
	for i in range(0, len(fff)):
		if(i%2==1): continue
		ff.append(fff[i])
	
	for i in range(0, len(ff)-3):
		x,y,z = sent_token(voc,ff[i]),sent_token(voc,ff[i+1]), sent_token(voc,ff[i+2])
		if(len(x)==0 or len(y)==0 or len(z)==0): continue
		pair.append((x,y,z))
	"""
	random.shuffle(pair)
	
	mtrain = bucket(pair[:(int)(len(pair)*.7)])
	mtest = bucket(pair[(int)(len(pair)*.7):])
	#bucket it
	pickle.dump( (mtrain, mtest), open( "data", "wb" ) )
	return mtrain, mtest

def readfnp():
	if(os.path.isfile('datanp')):
		return pickle.load( open( "datanp", "rb" ) )
	train, test = pickle.load( open( "data", "rb" ) )
	nptrain = {}
	nptest = {}
	for i in range(0, 4):
		nptrain[i] = np.array(train[i])
		nptest[i] = np.array(test[i])
	pickle.dump( (nptrain, nptest), open( "datanp", "wb" ) )
	return nptrain, nptest

def getnextr(size, train, i, idx):
#	i=random.randint(1, 3)
#	idx = np.random.choice(len(train[i]),size=size, replace=False)
	return train[i][idx]


def getnexte(size, test, i, idx):
#	i=random.randint(1, 3)
#	idx = np.random.choice(len(mtest[i]),size=size, replace=False)
	return mtest[i][idx]

if __name__ == "__main__":
	readf('chat.txt')
	print('done')
