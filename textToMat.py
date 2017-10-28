import numpy

def textToMat(string) :
    mat = numpy.zeros((59, 260))
    for i in range(0,len(string)):
        ascii = ord(string[i])

        #   a b c ... z = 0 1 2 ... 25
        if ((ascii > 96) and (ascii < 123)) :
            j = (ascii-97)
            mat[j][i] = 1

        #   <spc> ! " ... @ = 26 27 28 ... 58
        elif ((ascii > 31) and (ascii < 65)) :
            j = (ascii-6)
            mat[j][i] = 1

    return mat