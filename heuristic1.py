import csv 
import pandas
import json 
import operator

def preprocess():
	trainHandle = open('train.json').read()

	j = json.loads(trainHandle)
	#######################################

	unqCus = list()
	tempInList = list()
	try:
		count = 0 
		while True:
			a = j[count]['cuisine']
			if a not in unqCus:
				unqCus.append(a)

			count = count + 1
	except:
		print "unique got it"			


	#keeping the top 7 ingredients of each cuisine
	finalList = list()

	for cu in unqCus:
		tempDict = dict()
		try:
			cc = 0
			while True:
				b = j[cc]['ingredients']
				cus = j[cc]['cuisine']
				if cu == cus:
					for item in b:
						tempDict[item] = tempDict.get(item,0) + 1

				cc = cc + 1
				print 'this is cc',cc

		except:
			# print 'processing...'

			sorted_x = sorted(tempDict.items(), key=operator.itemgetter(1), reverse = True)
			breakCnt = 0
			for k,v in sorted_x:
				finalList.append(k)
				breakCnt = breakCnt + 1
				if breakCnt == 200:
					break
			print sorted_x		
			continue		

	# print finalList
	print len(finalList), len(unqCus)

	finalListClear = list()

	for li in finalList:
		if li not in finalListClear:
			finalListClear.append(li)

	print 'the final len is', len(finalListClear)
	return finalListClear
