import csv 
import pandas
import json 
from heuristic1 import preprocess
# from heuristic_weight import getSome

a = 'this is a a test string ok'
b = 'is this a test string for sure'
c = 'hello are you a sure this is a test message'

aList = a.split()
bList = b.split()
cList = c.split()

unq = list()



trainHandle = open('train.json').read()

testHandle = open('test.json').read()

j = json.loads(trainHandle)

jTest = json.loads(testHandle)
############################
# for line in aList:
# 	if line not in unq:
# 		unq.append(line)


# for line in bList:
# 	if line not in unq:
# 		unq.append(line)
		
# for line in cList:
# 	if line not in unq:
# 		unq.append(line)
		
# print unq

# mergedStr = a + b + c
unqing = list()

finalList = preprocess()
# try:
# 	count = 0
# 	while True:
# 		a = j[count]['ingredients']
# 		for iting in a:
# 			if iting not in unqing:
# 				unqing.append(iting)

# 		count = count + 1
# except:
# 	print 'done'
# 	c1 = 0
# 	for li in unqing:
# 		c1 = c1 + 1
# 		print li				

# print 'unq ingredients are', c1

# cusUnq = list()

# try:
# 	cc1 = 0
# 	while True:
# 		c = j[cc1]['cuisine']
# 		if c not in cusUnq:
# 			cusUnq.append(c)

# 		cc1 = cc1 + 1
# except:
# 	print cusUnq			

# cntDict = dict()

# try:
# 	cc1 = 0
# 	while True:
# 		d = j[cc1]['ingredients']
# 		for itm in d:
# 			cntDict[itm] = cntDict.get(itm,0) + 1
# 		cc1 = cc1 + 1
# except:
# 	print 'dict done'			

# q = 0
# for k,v in cntDict.items():
# 	if v > 250:
# 		q = q + 1
# 		print k,v
# 		unqing.append(k)
# print 'count is', q

##unquer 
# unqq = list()
# try:
# 	co1 = 0
# 	while True:
# 		a = j[cc]['cuisine']
# 		if not in unqq:
# 			unqq.append(a)

# 		co1 = co1 + 1
# except:
# 	print 'got unq'			


# for i in unqq:
# 	tupp = getSome(i)

# 	with open('SparseNewMahwash.csv', 'a') as nw:

# 		b = csv.writer(nw)
# 		# b.writerows(i) 
# 		try:
# 			list1 = list()
# 			list1.append(i)
# 			for k,v in tup:
# 				list1.append(k)

# 			b.writerows([list1])	

enn = list()

for li in finalList:
	li = li.encode('utf-8')
	enn.append(li)


# with open('xgboostTrain.csv', 'a') as nw:
# 	b = csv.writer(nw)
# 	b.writerows([enn])


# 	try:
# 		cc = 0
# 		while True:
# 			mList = list()
# 			a1 = j[cc]['ingredients']
# 			for itm in finalList:
# 				if itm not in a1:
# 					mList.append('0')
				
# 				else:
# 					mList.append('1')
				
# 			cc = cc + 1
# 			b.writerows([mList])

# 			# while len(mList) != 0:
# 			# 	mList.pop()
# 	except:
# 		print 'donee'


with open('testMah.csv', 'a') as nt:
	te = csv.writer(nt)
	te.writerows([enn]) 
	try:
		cc = 0
		while True:
			mList = list()
			a1 = jTest[cc]['ingredients']
			for itm in finalList:
				if itm not in a1:
					mList.append('0')
			
				else:
					mList.append('1')
			
			cc = cc + 1
			te.writerows([mList])

				# while len(mList) != 0:
				# 	mList.pop()
	except:
		print 'donee test'





		
	
					
