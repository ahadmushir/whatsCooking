#trying to predict the classification by the highest number match.. 
#first by taking the max ingredients for each cusine and then by finding the differences in the test 
import json 
import pandas
import csv 

trainHandle = open('train.json').read()

testHandle = open('test.json').read()

j = json.loads(trainHandle)

jTest = json.loads(testHandle)

dataTest = pandas.read_json('test.json')
############################
#getting unique cusines
unqCusines = list()
count1 = 0
try:
	while True:
		a = j[count1]['cuisine']
		if a not in unqCusines:
			unqCusines.append(a)
		count1 = count1 + 1

except:
	print unqCusines

##########################
check1 = list()
getIDMax = list()
try:
	for cus in unqCusines:
		count2 = 0
		small = -1
		try:
			while True:
				c = j[count2]['cuisine']

				if cus == c:
					ingL = j[count2]['ingredients']
					if len(ingL) > small:
						if len(check1) != 0:

							check1.pop() 

						small = len(ingL)
						idd = j[count2]['id']
						check1.append(idd)

				count2 = count2 + 1		

		except: 
			print cus, small
			getIDMax.append(idd)
			continue


except:
	print 'please chal ja'					

print check1
#got ID of toppers
print getIDMax


##############################
overallList = list()



for idz in getIDMax:
	c3 = 0
	try:
		while True:

			idd1 = j[c3]['id']
			idinIng = j[c3]['ingredients']
			idCus = j[c3]['cuisine']
			if idz == idd1:
				nameO = str(idCus)
				nameO = list()
				nameO.append(idCus)
				for itz in idinIng:
					nameO.append(itz)
				overallList.append(nameO)	
			c3 = c3 + 1		


	except: 
		continue		

#got the selected training tuples, all in a cute little list
print overallList[0][1:]


##############################
#print json.dumps(jTest, indent = 4)

alphaList = list()
with open('submission1.csv', 'a') as nw:
	try:
		testCounter = 0
		smallAgain = 0
		namee = 'None'
		while True:
			omegaList = list()
			tId = jTest[testCounter]['id']
			tIng = jTest[testCounter]['ingredients']
			
			c5 = 0
			while c5 != 20:
				tempList = list()
				for itt in tIng:
				
					checkingList = overallList[c5][1:]
					for isItThere in checkingList:
						if itt == isItThere:
							tempList.append(itt)


				if len(tempList) >= smallAgain:
					smallAgain = len(tempList)
					namee = str(overallList[c5][0])
				while len(tempList) != 0:
					tempList.pop()

				c5 = c5 + 1		

			omegaList.append(tId)
			omegaList.append(namee)
			alphaList.append([tId,namee])
			print omegaList	
			#if testCounter == 200:


			#	break

			#alphaList.append(omegaList)

			testCounter = testCounter + 1


			#b = csv.writer(nw) 
			#b.writerows(omegaList)

			while len(omegaList) != 0:
				omegaList.pop()


			


	except:
		print 'fingers crossed'
			
				
def enterInFile(Alist):
	checkinggstr = list()

	for itm in Alist:
		ii = str(itm)
		checkinggstr.append(ii)

	nw = open('submission1.csv', 'a') 

	b = csv.writer(nw) 
	b.writerows(checkinggstr)



for ch in alphaList:

	print ch

nw = open('submission1.csv', 'a')

b = csv.writer(nw) 
b.writerows(alphaList)

print len(alphaList)
print len(dataTest)
