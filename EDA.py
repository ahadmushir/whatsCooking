#The library used
import pandas
import json
import operator
import numpy as np
import matplotlib.pyplot as plt

#Opeining and reading training data
trainHandle = open('train.json').read()
#Since pandas provide more analytical feature so reading by pandas too 
data = pandas.read_json('train.json')

#loading json data
j = json.loads(trainHandle)
# print j[0]

#Using describe function to get overall analyses
print 'overall description',data.describe()
#Showing total rows
print 'total rows of train data',len(data)
#Showing Columns
print 'total columns', len(data.columns)
#Printing all the columns
print data.columns

####
#For geting different type of cuisines
unqlist = list()

try:
	count = 0
	while True:
		a = j[count]['cuisine']
		for item in a:
			if item not in unqlist:
				unqlist.append(item)
		count = count + 1
except:
	print 'done'

print 'different types of cuisine present',len(unqlist)

####
#For getting different types of ingredients
unqlist1 = list()
try:
	count = 0
	while True:
		a = j[count]['ingredients']
		for item in a:
			if item not in unqlist1:
				unqlist1.append(item)
		count = count + 1
except:
	print 'done'

print 'Unique no. of ingredients present',len(unqlist1)

####
#For getting the top 10 most occuring ingredients
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


print type(sorted_x)
# for k,v in sorted_x:
# 	c = 0

# 	if v > 200:
# 		print k,v	
# 	# break	
# 	if c == 1:
# 		break	
# 	c = c + 1

ingList = list()
valList = list()

for i,v in sorted_x:
	if v > 4000:

		print i,v
		ingList.append(i)
		valList.append(v)

		if len(ingList) and len(valList) == 10:
			break


print len(ingList),len(valList)
print ingList[len(ingList)-1],valList[len(valList)-1]

####
#For generating a graph showing the most occurred ingredients using matplotlib lib

width = 1
x = range(0,len(ingList))
x1 = range(0,len(ingList) + 1)
plt.bar(x,valList, width, color="blue")
plt.title('Initial Analyses')
plt.ylabel('frequency')
plt.xlabel('Top 10 ingredients')
#plt.xticks(bar_width, ('A', 'B', 'C', 'D'))
plt.xticks(x1, ingList, rotation = 45, size = 8.5)
fig = plt.gcf()
plt.show()

