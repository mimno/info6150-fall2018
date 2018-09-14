import re, sys, numpy
from collections import Counter

"""
In this script I'm implementing an EM algorithm for the "asymmetric" version of
pLSI as presented in the Hofmann paper.
"""

doc_counters = []
corpus_counts = Counter()
document_frequency = Counter()

num_topics = 20
num_iterations = 100

doc_text = []
print ("reading")

with open(sys.argv[1], encoding="utf-8") as reader:
    for line in reader:
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            tag = fields[1]
            tokens = fields[2].lower().split()
            
            doc_counter = Counter(tokens)
            corpus_counts.update(doc_counter)
            document_frequency.update(doc_counter.keys())
            doc_counters.append(doc_counter)
            
            doc_text.append(fields[2])

num_docs = len(doc_counters)

## construct a vocabulary list in reverse order by corpus count
vocabulary = [ w for w, c in corpus_counts.most_common() if c > 5 and document_frequency[w] < 0.1 * num_docs ]
reverse_vocab = { word:i for (i, word) in enumerate(vocabulary) }
vocab_size = len(vocabulary)

## Remove pruned low-frequency words from the vocabulary
for doc_counter in doc_counters:
    pruned_words = [ w for w in doc_counter.keys() if w not in reverse_vocab ]
    
    for word in pruned_words:
        del doc_counter[word]

doc_counters = [ c for c in doc_counters if len(c) > 0 ]

num_docs = len(doc_counters)

## Initialization! This is important. Everything else after this 
##  point is deterministic. Use a fairly flat Dirichlet prior.

doc_dirichlet_prior = 10.0 * numpy.ones(num_topics)
word_dirichlet_prior = 10.0 * numpy.ones(vocab_size)

## for docs the distribution is over topics
current_doc_topics = numpy.random.dirichlet(doc_dirichlet_prior, num_docs)

## columns are distributions, so transpose the word distributions
current_word_topics = numpy.random.dirichlet(word_dirichlet_prior, num_topics).T

def iterate():
    
    ## The model log likelihood is a good indicator of progress in training.
    ##  It should always be negative and *NEVER* decrease -- if those aren't true,
    ##  there is a bug.
    model_log_likelihood = 0.0
    
    ## We will hold the current model fixed and re-estimate a new model
    ## using the current values. We therefore need to allocate space for
    ## this new model.
    new_doc_topics = numpy.zeros((num_docs, num_topics))
    new_word_topics = numpy.zeros((vocab_size, num_topics))
    
    for doc_id, doc_counter in enumerate(doc_counters):
        
        for word in doc_counter.keys():
            if not word in reverse_vocab:
                continue
            
            word_id = reverse_vocab[word]
            count = doc_counter[word]
            
            ## This word occurs `count` times in the document.
            ## We want to assign some fraction of those tokens to each topic.
            
            #print(current_doc_topics[doc_id,:])
            
            ## Get the expectation of the posterior probability of each
            ##  topic for this word, and divide the probability.
            expectation = current_doc_topics[doc_id,:] * (current_word_topics[word_id,:])
            
            sum_expectation = numpy.sum(expectation)
            model_log_likelihood += count * numpy.log(sum_expectation)
            
            expectation /= sum_expectation  ## normalize
            expectation *= count ## scale up to the token count
            
            new_doc_topics[doc_id,:] += expectation
            new_word_topics[word_id,:] += expectation
    
    ## Normalize the document distributions along rows
    row_sums = numpy.sum(new_doc_topics, axis=1)
    new_doc_topics /= row_sums[:, numpy.newaxis]
    
    ## Normalize the topic distributions along columns
    col_sums = numpy.sum(new_word_topics, axis=0)
    new_word_topics /= col_sums[numpy.newaxis, :]
    
    return (model_log_likelihood, new_doc_topics, new_word_topics)

def rank_words(x):
    return sorted(zip(x, vocabulary), reverse=True)

def rank_docs(x):
    return sorted(zip(x, doc_text), reverse=True)

def print_top_words():
    for topic in range(num_topics):
        print(" ".join([w for c,w in rank_words(current_word_topics[:,topic])[:20]]))

for iteration in range(num_iterations):
    ll, new_doc, new_word = iterate()
    current_doc_topics = new_doc
    current_word_topics = new_word
    if ((iteration+1) % 5 == 0):
        print_top_words()
    print(ll)
