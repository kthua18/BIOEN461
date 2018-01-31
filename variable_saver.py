import pickle
import numpy as np


def save_function(a, b, c, d, e):
   fileName = "values"
   valueArray = np.array([a, b, c, d, e], dtype=object)
   fileObject = open(fileName, 'wb')
   pickle.dump(valueArray, fileObject)
   fileObject.close()