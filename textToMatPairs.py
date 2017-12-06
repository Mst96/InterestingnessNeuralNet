import numpy
import sys

def textToMatPairs(string) :
    mat = [0] * (26*26)
    for i in range(0,(len(string)-1)):

        firstLetter = ord(string[i])
        secondLetter = ord(string[i+1])

        if((firstLetter > 96) and (firstLetter < 123) and (secondLetter > 96) and (secondLetter < 123)):

            row = (firstLetter-97)

            column = (secondLetter - 97)

            mat[column * row] += 1
    return mat