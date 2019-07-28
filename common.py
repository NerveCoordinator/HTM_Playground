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

# generates a scalar encoder for numbers
# between mini and maxi of specified size and sparsity
def ScalarEncoderGenerator(mini, maxi, size, spars=-1):
    # if sparsity is not given, don't allow overlapping bits.
    if spars == -1:
        # unsure if correct
        diff = maxi-mini
        spars = 1/(diff+1)         
    params            = ScalarEncoderParameters()
    params.minimum    = mini
    params.maximum    = maxi
    params.size       = size
    params.sparsity   = spars
    encoder = ScalarEncoder( params )
    return encoder

# from sp tutorial
# Utility routine for printing an SDR in a particular way.
def formatBits(sdr):
  s = ''
  for c in range(sdr.size):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(sdr.dense.flatten()[c])
  s += ' '
  return s

# Deleet's list_flatten from  https://stackoverflow.com/a/40547477
def flatten(l, a=None):
    #check a
    if a is None:
        #initialize with empty list
        a = []

    for i in l:
        if isinstance(i, list):
            flatten(i, a)
        else:
            a.append(i)
    return a


# takes list of SDRs and returns them as a single
# SDR. Also returns the width for convenience.
def combineBits(bitList):
    # if it is a list of lists, make 1D 
    bits = flatten(bitList)  
    encodingWidth = 0
    for encoding in bits:
        encodingWidth += encoding.size    
    combined = SDR( encodingWidth ).concatenate(bits)
    return encodingWidth, combined


# asks classifier what the SDR probably is. 
def decode(decoder, encoded):
    return numpy.argmax(decoder.infer( encoded ) )

# trains a classifier on numbers between a range        
def trainNumDecoder(encoder, mini, maxi, noise):
    clsr = Classifier()
    # loop from smallest to largest number 10 times
    for y in range(10):
        for x in range(mini,maxi+1):
            # encode current number
            encoded = encoder.encode(x)
            #corruptSDR(encoded, noise) 
            # associate encoding to 'class' 
            # but classes here are numbers.
            clsr.learn(encoded, x)
    # test every number to see if we learned it
    for x in range(mini,maxi+1):
        encoded = encoder.encode(x)
        out = decode(clsr,encoded)
        if out != x:
            print("error in decode training:", out, "->", x)
    return clsr

# TM is given an input big enough to hold
# both question and answer, but we only show
# it one at a time. Since TM predicts its next
# input we can use this to do computations.
def promptTM(tm, question, answer, learn=True):  

    # we feed it the question
    tm.compute(question, learn=learn)  
    # what does it expect the answer will be?
    tm.activateDendrites(True)
    prediction = tm.getPredictiveCells() 
    if learn: 
        # show it the answer
        tm.compute(answer,learn=learn)  
        # reset it so it doesn't associate
        # this answer to the next question
        tm.reset()
    return prediction

'''
# from sp tutorial
# not actually used yet
def corruptSDR(sdr, noiseLevel):
      """
      Corrupts a binary vector by inverting noiseLevel percent of its bits.
    
      Argument vector     (array) binary vector to be corrupted
      Argument noiseLevel (float) amount of noise to be applied on the vector.
      """
      vector = sdr.flatten().dense
      for i in range(sdr.size):
        rnd = random.random()
        if rnd < noiseLevel:
          if vector[i] == 1:
            vector[i] = 0
          else:
            vector[i] = 1
      sdr.dense = vector
'''   
'''
I want to calculate sparcity for a given number
of duplicate and unique bits per 'unit'. That is, if I
have 7 categories and I want 5 unique bits per category,
then how sparse should it be given that I want it
to be X bits total?

# ??? unsure on correctness
def desiredSparsity(digits, size, bits):
    return digits / (float(size) * bits)


closer to correct - for digits 0 - 9
algorithm for calculating number of unique bits per digit
decBits        = 3
decDupBits     = 2 # per side
decSize        = decBits * 10 + decDupBits
decUniqueBits  = (decSize - decDupBits * 10 )/10
print("# unique:",decUniqueBits)
decSpars       = (decBits+decDupBits) / float(decSize)

but need to reverse the decUniqueBits equation and so we
can specify how many unique bits we want. I think you can
only specify two of [decSize, decDupBits, and decUniqueBits]
but need to think that through some more.

'''
