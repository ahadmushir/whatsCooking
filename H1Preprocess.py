#Heuristic for getting 
####
#Importing lib
import csv 
import pandas
import json 

unq = list()
####
#Loading data 
trainHandle = open('train.json').read()

testHandle = open('test.json').read()

j = json.loads(trainHandle)

jTest = json.loads(testHandle)

####
#For finding the counts of all ingredients

unqing = list()
finalList = list()
unqDict = dict()

try:
	countDict = 0
	while True:
		b = j[countDict]['ingredients']
		for item in b:
			unqDict[item] = unqDict.get(item,0) + 1

		countDict = countDict + 1
		
except:
	print 'ingredients count done!!'
	sorted_x = sorted(unqDict.items(), key=operator.itemgetter(1), reverse = True)

####
#For getting top 200 ingredients in the list
ccc = 0
for i,v in sorted_x:

	finalList.append(i)
	if ccc = 200:
		break
	ccc = ccc + 1	

enn = list()

####
#For encoding to get the columns names in utf8 format
for li in finalList:
	li = li.encode('utf-8')
	enn.append(li)

####
#Making the sparse matrix by placing 1 and 0

with open('TrainHeuristic1.csv', 'a') as nw:
	b = csv.writer(nw)
	b.writerows([enn])


	try:
		cc = 0
		while True:
			mList = list()
			a1 = j[cc]['ingredients']
			for itm in finalList:
				if itm not in a1:
					mList.append('0')
				
				else:
					mList.append('1')
				
			cc = cc + 1
			b.writerows([mList])

			# while len(mList) != 0:
			# 	mList.pop()
	except:
		print 'donee'
