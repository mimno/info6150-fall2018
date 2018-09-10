import re, sys, numpy
from collections import Counter
from scipy.sparse import lil_matrix
import scipy.sparse.linalg

doc_counters = []
corpus_counts = Counter()

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
            doc_counters.append(doc_counter)
            
            doc_text.append(fields[2])

num_docs = len(doc_counters)

## construct a vocabulary list in reverse order by corpus count
vocabulary = [ w for w, c in corpus_counts.most_common() if c > 5 ]
reverse_vocab = { word:i for (i, word) in enumerate(vocabulary) }
vocab_size = len(vocabulary)

print("constructing matrix")
doc_word_counts = lil_matrix((num_docs, vocab_size))

for doc_id, doc_counter in enumerate(doc_counters):
    words = list([word for word in doc_counter if word in reverse_vocab])
    counts = [doc_counter[word] for word in words]
    word_ids = [reverse_vocab[word] for word in words]
    
    doc_word_counts[doc_id,word_ids] = counts

doc_word_counts = doc_word_counts.tocsr()

print("running SVD")
doc_vectors, singular_values, word_vectors = scipy.sparse.linalg.svds(doc_word_counts, 100)

word_vectors = word_vectors.T

def rank_words(x):
    return sorted(zip(x, vocabulary))

def rank_docs(x):
    return sorted(zip(x, doc_text))

def l2_norm(matrix):
    row_norms = numpy.sqrt(numpy.sum(matrix ** 2, axis = 1))
    return matrix / row_norms[:, numpy.newaxis]

