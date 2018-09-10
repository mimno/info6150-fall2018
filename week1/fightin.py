import re, sys, numpy
from collections import Counter


ds_counts = Counter()
with open(sys.argv[1], encoding="utf-8") as reader:
    for line in reader:
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            tag = fields[1]
            tokens = fields[2].lower().split()
            
            ds_counts.update(tokens)


stats_counts = Counter()
with open(sys.argv[2], encoding="utf-8") as reader:
    for line in reader:
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            tag = fields[1]
            tokens = fields[2].lower().split()
            
            stats_counts.update(tokens)

corpus_counts = ds_counts + stats_counts
vocabulary = list(corpus_counts.keys())
reverse_vocab = { word:i for (i, word) in enumerate(vocabulary) }
vocab_size = len(vocabulary)

y_ds = numpy.zeros(vocab_size)
y_stats = numpy.zeros(vocab_size)
y_corpus = numpy.zeros(vocab_size)

for i, word in enumerate(vocabulary):
    y_ds[i] = ds_counts[word]
    y_stats[i] = stats_counts[word]
    y_corpus[i] = corpus_counts[word]

n_ds = sum(y_ds)
n_stats = sum(y_stats)
n_corpus = sum(y_corpus)

smoothing = 100.0
f_ds = (y_ds + smoothing) / (n_ds + smoothing * vocab_size)
f_stats = (y_stats + smoothing) / (n_stats + smoothing * vocab_size)

odds_ds = f_ds / (1.0 - f_ds)
odds_stats = f_stats / (1.0 - f_stats)

#diff_y = (y_ds / n_ds) - (y_stats / n_stats)
diff_y = numpy.log(f_ds / f_stats)
sorted_words = sorted(zip(diff_y, vocabulary))
print(sorted_words[:10])
print(sorted_words[-10:])

