import ujson, sys, numpy
import matplotlib
import matplotlib.pyplot as pyplot

sentences = []

with open(sys.argv[1]) as infile:
    for line in infile:
        sentences.append(ujson.loads(line))

tokens = []
vector_buffer = []

for sentence in sentences:
    for token_data in sentence['features']:
        tokens.append(token_data['token'])
        vector_buffer.append(numpy.array(token_data['layers'][0]['values']))
        
        ## For printing out vectors in space-delimited form
        #vector = ["{:.3f}".format(x) for x in token_data['layers'][0]['values']]
        # print("{} {}".format(token, " ".join(vector)))

token_vectors = numpy.array(vector_buffer)

def indices_of(s):
    return [ i for i, w in enumerate(tokens) if w == s ]

def nearest(i):
    sorted_tokens = sorted(zip(token_vectors.dot(token_vectors[i,:]), tokens), reverse=True)
    [s for x, s in sorted_tokens[:100]]

def hist(x):
    n, bins, patches = pyplot.hist(x, 50, density=True)
    pyplot.show()