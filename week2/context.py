import sys, re, math
from collections import Counter

all_words = Counter()
word_contexts = {}

with open(sys.argv[1]) as infile:
    for line in infile:
        fields = line.rstrip().split("\t")
        
        if len(fields) != 3:
            continue
        
        tokens = fields[2].split(" ")
        
        all_words.update(tokens)
        
        for i in range(len(tokens)):
            word = tokens[i]
            
            if not word in word_contexts:
                word_contexts[word] = Counter()
            
            if i > 0:
                word_contexts[word][tokens[i-1]] += 1
            if i + 1 < len(tokens):
                word_contexts[word][tokens[i+1]] += 1

def similarity(a, b):
    
    overlap = set(a.keys()) & set(b.keys())
    a_norm = 1.0 / sum(a.values())
    b_norm = 1.0 / sum(b.values())
    
    ## start by assuming there is no overlap
    diff = 2.0
    
    for word in overlap:
        a_prob = a[word] * a_norm
        b_prob = b[word] * b_norm
        diff += abs(a_prob - b_prob) - a_prob - b_prob
    
    return 1.0 - diff / 2

## Initialize clusters
cluster_counts = []
cluster_words = []

for word, n in all_words.most_common(5000):
    if len(cluster_counts) < 200:
        cluster_counts.append(word_contexts[word].copy())
        cluster_words.append([word])
    else:
        best_cluster = -1
        best_similarity = 0.0
        for cluster_id in range(len(cluster_counts)):
            current_similarity = similarity(cluster_counts[cluster_id], word_contexts[word])
            if current_similarity > best_similarity:
                best_cluster = cluster_id
                best_similarity = current_similarity
        
        cluster_counts[best_cluster] += word_contexts[word]
        print("{:.3f}\tmerging {} with {}".format(best_similarity, word, " ".join(cluster_words[best_cluster])))
        cluster_words[best_cluster].append(word)

for cluster in cluster_words:
    print(" ".join(cluster))
