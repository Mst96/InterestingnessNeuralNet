import numpy

def textToMatPairs(string) :
    mat = numpy.zeros((26, 26))
    for i in range(0,(len(string)-1)):

        asciiFirstPartOfPair = ord(string[i])
        asciiSecondPartOfPair = ord(string[i+1])

        if ((asciiFirstPartOfPair > 96) and (asciiFirstPartOfPair < 123)) :
            row = (asciiFirstPartOfPair-97)

        if ((asciiSecondPartOfPair > 96) and (asciiSecondPartOfPair < 123)):
            collumn = (asciiSecondPartOfPair - 97)

        mat[collumn][row] += 1

    print(mat)