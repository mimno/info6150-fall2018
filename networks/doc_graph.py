import sys, re, math
from collections import Counter
from scipy.stats import poisson

all_words = Counter()
word_contexts = {}

stopwords = set("the to a of I is and in you that for itdata this have be with are can on as not".split())

corpus_size = 0
sum_squared_lengths = 0

row_num = 0

with open(sys.argv[1]) as infile:
    for line in infile:
        fields = line.rstrip().split("\t")
        
        if len(fields) != 3:
            continue
        
        row_num += 1
        #if row_num > 1000:
        #    break
        
        #tokens = fields[2].split(" ")
        tokens = sorted([s for s in set(fields[2].split()) if len(s) > 1 and s not in stopwords])
        
        all_words.update(tokens)
        doc_length = len(tokens)
        corpus_size += doc_length
        sum_squared_lengths += doc_length * doc_length
        
        for i in range(len(tokens) - 1):
            left_word = tokens[i]
            
            if not left_word in word_contexts:
                word_contexts[left_word] = Counter()
                
            for j in range(i+1, len(tokens)):
                right_word = tokens[j]
                
                word_contexts[left_word][right_word] += 1

poisson_rate_coefficient = sum_squared_lengths / (corpus_size * corpus_size)

min_rejected_rate = {}

for left_word in word_contexts:
    if not all_words[left_word] > 10:
        continue
    for right_word in word_contexts[left_word]:
        if not all_words[right_word] > 10:
            continue
        
        original_count = word_contexts[left_word][right_word]
        poisson_rate = all_words[left_word] * all_words[right_word] * poisson_rate_coefficient
        
        if original_count in min_rejected_rate and poisson_rate > min_rejected_rate[original_count]:
            continue
        
        cutoff = poisson.ppf(0.95, poisson_rate)
        
        count = original_count - cutoff
        if count > 0:
            #print("{}\t{}\t{}\t{}\t{}\t{}".format(left_word, right_word, word_contexts[left_word][right_word], poisson_rate, cutoff, count))
            print("{}\t{}\t{}".format(left_word, right_word, int(count)))
        else:
            min_rejected_rate[original_count] = poisson_rate
