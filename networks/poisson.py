import numpy, sys

num_clusters = int(sys.argv[2])

vocabulary = [] # int to string
reverse_vocabulary = {} # string to int

## This function builds a vocabulary as we see each symbol
def get_id(s):
    if s in reverse_vocabulary:
        return reverse_vocabulary[s]
    else:
        symbol_id = len(vocabulary)
        reverse_vocabulary[s] = symbol_id
        vocabulary.append(s)
        return symbol_id

## Edges will be a list of (left ID, right ID, count) tuples
edges = []

## load the network edges
with open(sys.argv[1], encoding="utf-8") as network_file:
    for line in network_file:
        if line.startswith("#"):
            continue
        
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            left_id = get_id(fields[0])
            right_id = get_id(fields[1])
            count = int(fields[2])
            
            edges.append( (left_id, right_id, count) )

num_symbols = len(vocabulary)

## Use a symmetric Dirichlet to initialize cluster weights for each
##  symbol to close to uniform
symbol_cluster_weights = numpy.random.dirichlet( numpy.ones(num_clusters), num_symbols )
## Calculate the column square roots

symbol_cluster_buffer = numpy.zeros( ( num_symbols, num_clusters) )

def top_symbols(cluster):
    return sorted(zip(symbol_cluster_weights[:,cluster], vocabulary), reverse=True)[:20]

def display():
    for cluster in range(num_clusters):
        print(" ".join([ "{} ({:.2f})".format(name, score) for score, name in top_symbols(cluster)]))

for iteration in range(30):
    symbol_cluster_buffer = numpy.zeros( ( num_symbols, num_clusters) )
    iteration_sum = 0
    
    for left_id, right_id, count in edges:
        ## elementwise product
        product = symbol_cluster_weights[left_id,:] * symbol_cluster_weights[right_id,:]
        row_sum = numpy.sum(product)
        iteration_sum += row_sum
        
        if row_sum > 0.0:
            update = count * product / row_sum
            symbol_cluster_buffer[left_id,:] += update
            symbol_cluster_buffer[right_id,:] += update
        else:
            print("zero edge: {} {} {}".format(left_id, right_id, count))
    
    print(iteration_sum)
    
    cluster_square_roots = numpy.sqrt(numpy.sum(symbol_cluster_buffer, axis=0))
    symbol_cluster_weights = symbol_cluster_buffer / cluster_square_roots[numpy.newaxis, :]

display()
    
