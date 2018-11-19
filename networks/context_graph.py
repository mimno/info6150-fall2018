import sys, re, math
from collections import Counter

all_words = Counter()
word_contexts = {}

stopwords = set("the to a of I is and in you that for itdata this have be with are can on as not".split())

with open(sys.argv[1]) as infile:
    for line in infile:
        fields = line.rstrip().split("\t")
        
        if len(fields) != 3:
            continue
        
        #tokens = fields[2].split(" ")
        tokens = [s for s in fields[2].split() if s not in stopwords]
        
        all_words.update(tokens)
        
        for i in range(len(tokens)):
            word = tokens[i]
            
            if not word in word_contexts:
                word_contexts[word] = Counter()
            
            if i > 0:
                word_contexts[word][tokens[i-1]] += 2
            if i + 1 < len(tokens):
                word_contexts[word][tokens[i+1]] += 2

            if i > 1:
                word_contexts[word][tokens[i-2]] += 1
            if i + 2 < len(tokens):
                word_contexts[word][tokens[i+2]] += 1

seen_pairs = set()

for left_word in word_contexts:
    if not all_words[left_word] > 10:
        continue
    for right_word in word_contexts[left_word]:
        if not all_words[right_word] > 10:
            continue

        if "{}\t{}".format(right_word, left_word) in seen_pairs:
            continue
        
        seen_pairs.add("{}\t{}".format(left_word, right_word))
        print("{}\t{}\t{}".format(left_word, right_word, word_contexts[left_word][right_word]))
