from operator import itemgetter
from itertools import groupby

import sys



file = open("/Users/Astrid/Downloads/pg63032.txt", "rt")

wordcount = {}

for line in file:
    line = line.strip()
    word, count = line.split('\t', 1)
    try:

        count = int(count)
    except ValueError:
        continue

    try:
        wordcount[word] = wordcount[word]+count
    except:
        wordcount[word] = count


for word in wordcount.keys():
    print ('%s\t%s'% ( word, wordcount[word] ))