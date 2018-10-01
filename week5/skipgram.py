import sys, re, numpy, random
from collections import Counter

word_pattern = re.compile(r"\w+")

num_dimensions = int(sys.argv[2])
window_size = 5
num_negative_samples = 5
frequency_weight = 0.0001

raw_documents = []
all_words = Counter()

with open(sys.argv[1]) as infile:
    for line in infile:
        fields = line.rstrip().split("\t")
        
        if len(fields) != 3:
            continue
        
        tokens = word_pattern.findall(fields[2].lower())
        
        all_words.update(tokens)
        raw_documents.append(tokens)

## construct a vocabulary list in reverse order by corpus count
vocabulary = [ w for w, c in all_words.most_common() if c > 5 ]
reverse_vocab = { word:i for (i, word) in enumerate(vocabulary) }
vocab_size = len(vocabulary)

total_words = sum(all_words.values())

retention_probability = numpy.zeros(vocab_size)
sampling_distribution = numpy.zeros(vocab_size)

for word_id, word in enumerate(vocabulary):
    word_probability = all_words[word] / total_words
    frequency_score = word_probability / frequency_weight
    
    ## Drop frequent words with this probability
    retention_probability[ word_id ] = min(1.0, (numpy.sqrt(frequency_score) + 1) / frequency_score)
    
    ## Generate a cumulative probability distribution for sampling negative words.
    if word_id == 0:
        sampling_distribution[ word_id ] = word_probability ** 0.75
    else:
        sampling_distribution[ word_id ] = word_probability ** 0.75
        ## Make it cumulative:
        sampling_distribution[ word_id ] += sampling_distribution[ word_id-1 ]

## Now renormalize so that the last element is 1.0
sampling_distribution /= sampling_distribution[-1]

## Create an index into the sampling distribution
centile_lookup = []
centile_word_id = 0
for centile in range(100):
    centile_target = centile * 0.01
    
    while sampling_distribution[centile_word_id] < centile_target:
        centile_word_id += 1
    
    print("centile {} word {} {}".format(centile, centile_word_id, vocabulary[centile_word_id]))
    centile_lookup.append(centile_word_id)

def sample_word():
    sample = random.random()
    ## Use the lookup table to start in a nearby place
    sampled_word_id = centile_lookup[ int(numpy.floor(sample * 100)) ]
    
    while sample > sampling_distribution[sampled_word_id]:
        sampled_word_id += 1
    
    return sampled_word_id

## Now allocate memory for the embeddings and counter-embeddings
embeddings = numpy.random.normal(0, 0.1, (vocab_size, num_dimensions))
counter_embeddings = numpy.random.normal(0, 0.1, (vocab_size, num_dimensions))

learning_rate = 0.025



def nearest(word, n):
    normalized_embeddings = l2_norm(embeddings)
    word_id = reverse_vocab[word]
    word_scores = rank_words(normalized_embeddings.dot(normalized_embeddings[word_id,:]))
    
    print(", ".join([ "{:.3f} {}".format(score, word) for score, word in word_scores[:n]]))

def rank_words(x):
    return sorted(zip(x, vocabulary), reverse=True)

def l2_norm(matrix):
    row_norms = numpy.sqrt(numpy.sum(matrix ** 2, axis = 1))
    return matrix / row_norms[:, numpy.newaxis]

def train():

    words_so_far = 0
    display_interval = 10000
    
    total_loss = 0.0
    
    for tokens in raw_documents:
        
        word_ids = [ reverse_vocab[t] for t in tokens if t in reverse_vocab ]
        
        ## subsample frequent words
        word_ids = [ word_id for word_id in word_ids if random.random() < retention_probability[word_id] ]
        
        for position in range(len(word_ids)):
            
            word = word_ids[position]
            
            effective_window_size = random.randint(1, window_size)
            
            start = max(0, position - effective_window_size)
            end = min(position + effective_window_size + 1, len(word_ids))
            
            gradient = numpy.zeros(num_dimensions)
            
            for context_position in range(start, end):
                if context_position == position:
                    continue
                
                context = word_ids[context_position]
                
                inner_product = embeddings[word,:].dot( counter_embeddings[context,:] )
                prediction = 1.0 / (1.0 + numpy.exp(-inner_product))  ## sigmoid function
                update = learning_rate * (1.0 - prediction)
                total_loss += update
                
                gradient += update * counter_embeddings[context,:]
                counter_embeddings[context,:] += update * embeddings[word,:]
                
                for s in range(num_negative_samples):
                    
                    context = sample_word()
                
                    inner_product = embeddings[word,:].dot( counter_embeddings[context,:] )
                    prediction = 1.0 / (1.0 + numpy.exp(-inner_product))  ## sigmoid function
                    update = (0.0 - prediction) * learning_rate
                    total_loss -= update
                
                    gradient += update * counter_embeddings[context,:]
                    counter_embeddings[context,:] += update * embeddings[word,:]
            
            embeddings[word,:] += gradient
            
            words_so_far += 1
            if words_so_far % display_interval == 0:
                print("{}\t{}".format(words_so_far, total_loss / display_interval))
                total_loss = 0.0
                nearest("regression", 10)
