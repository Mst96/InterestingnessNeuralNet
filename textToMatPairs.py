import numpy
import sys

def textToMatPairs(string) :
  mat = numpy.zeros((26,26))

  highest = 0
  for i in range(0,(len(string)-1)):
    firstLetter = ord(string[i])
    secondLetter = ord(string[i+1])

    if((firstLetter > 96) and (firstLetter < 123) and (secondLetter > 96) and (secondLetter < 123)):

        row = (firstLetter-97)

        column = (secondLetter - 97)

        mat[column][row] += 1
        if mat[column][row] > highest:
        	highest = mat[column][row]
  # mean = numpy.mean(mat)
  # sd = numpy.std(mat)
  mat = normalise(mat, highest)
  return mat


def normalise(mat, highest):
	for i, list in enumerate(mat):
		for j, item in enumerate(list):
			if(highest > 0):
				mat[i][j] = mat[i][j]/highest
	mat = mat.tolist()
	mat = [item for sublist in mat for item in sublist]
	return mat

def standardise(mat, mean, sd):
	for i, list in enumerate(mat):
		for j, item in enumerate(list):
			mat[i][j] = (mat[i][j] - mean)/sd
	mat = mat.tolist()
	mat = [item for sublist in mat for item in sublist]
	return mat


	def mean(mat):
		total = 0
		for i, list in enumerate(mat):
			for j, item in enumerate(list):
				total += mat[i][j]
		return total/676