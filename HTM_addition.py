####################################
# Prediction of addition using HTM #
####################################

import numpy
import pickle
import random
import sys
import tempfile
import unittest

from htm.encoders.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.algorithms import Classifier, Predictor
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

# various functions I use a lot
import common

# settings
numSize  = 200    # number of bits in SDR 
numMin   = 0      # smallest number we're encoding
numMax   = 100    # largest number we're encoding
numSpars = 0.02   # SDR sparcity
tries    = 50000 # to make sure we've given it enough examples. Mostly works with fewer than this.

# for converting numbers into SDRs as specified above
numEncoder = common.ScalarEncoderGenerator(numMin,numMax,numSize,numSpars)
# used to predict addition
tm = TM(columnDimensions = (numSize*3,),
      cellsPerColumn=1,
      initialPermanence=0.5,
      connectedPermanence=0.5,
      minThreshold=8,
      maxNewSynapseCount=20,
      permanenceIncrement=0.1,
      permanenceDecrement=0.0,
      activationThreshold=8,
      )  
# null SDR for padding
emptyBits = SDR(numSize)
# we store predictions in this
results = []

# generate a bunch of addition examples
for x in range (0,tries):
    # get two random numbers and sum them
    a = random.randint(0,numMax/2)
    b = random.randint(0,numMax/2)
    c = a + b 
    print(a, "+", b, "=", c)    
    # SDR for each number
    aBits = numEncoder.encode(a)
    bBits = numEncoder.encode(b)
    cBits = numEncoder.encode(c)    
    # combine SDRs 
    # question only shows a and b
    # answer only shows c
    width, question = common.combineBits([aBits, bBits, emptyBits])
    width, answer   = common.combineBits([emptyBits,emptyBits, cBits])    
    # TM predicts answer given question
    prediction = common.promptTM(tm, question, answer, True)    
    #if we've done half the examples, start recording answers
    if x > tries/2:
        results.append((c, prediction))
      
# Converts SDRs back into numbers
numDecoder = common.trainNumDecoder(numEncoder, numMin, numMax, 0)
# to count addition errors
errors = 0
err_list =[]
# Decode all predictions, see if they match.
# Working on a more elegant way to do this. Very messy.
for result in results:
    # ground truth
    val = result[0]     
    # bit array of our prediction
    raw_prediction = result[1].coordinates[0]

    if len(raw_prediction) > 0:
        # make blank SDR big enough for one number
        prediction = SDR(numSize)        
        # remove prompt blank space from answer
        for value in raw_prediction:
            prediction.dense[value-numSize*2] = 1            
        # tell SDR we updated its values 
        # (unsure why this works, found in sp tutorial)
        prediction.dense = prediction.dense         
        # convert prediction into a number!
        prediction = common.decode(numDecoder,prediction)
    else:
        prediction = None # no prediction
    # is prediction correct?
    agreement = (val == prediction)
    print("truth:", val, "prediction:", prediction, "agree?", agreement)
    if not agreement:
        errors += 1
        err_list.append((val,prediction))

print(err_list)

# should be 0 at default settings.
print("errors:", errors , "/", int(tries/2))


### misc info ###

'''
you can easily test sparsity with only a few inputs

if sparsity is too high
    it'll predict too many numbers and be way off

if it won't give any answer at all
    you may need to feed it more inputs
    or sparsity may be too low
    
if numSize is too low
    it'll give consistently wrong outputs for specific numbers
    often numbers near the start like 0 or 1
    this is because you don't have enough bits to encode.
    
'''

'''
# takes a while to teach, but can sum up to 250
numSize = 500
numMin  = 0
numMax  = 250
numSpars = 0.008
tries = 100000
'''

''' 
#decoder test
test = numEncoder.encode(8)
result = decode(numDecoder,test)
print(8,result)
'''
