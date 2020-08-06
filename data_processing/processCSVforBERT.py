'''
Converts data with POS tags and labels into train/dev/test.tmp.txt files.
'''

import csv

tokens = []
tags = []
sentence_ids = []
maxlen = 0 # longest sequence of words

with open('../data/dataWithPOS.csv') as csvfile:
    csvreader = csv.reader(csvfile)

    #list of all the fields
    fields = csvreader.__next__()
    print(fields)
    word_index = fields.index('Word')
    tag_index = fields.index('Tag')

    for row in csvreader:
        if len(row) == 0:
            continue
        tokens.append(row[word_index])
        tags.append(row[tag_index])
        sentence_ids.append(row[0])

print(tokens[:10], len(tokens))
print(tags[:10], len(tags))
print(sentence_ids[:10], len(sentence_ids))

# write to new .txt.tmp data files
trainf = open('../data/train.txt.tmp', 'w')
devf = open('../data/dev.txt.tmp', 'w')
testf = open('../data/test.txt.tmp', 'w')

dev_start_index = int(len(tokens) * 0.8)
test_start_index = int(len(tokens) * 0.9)

index = 0
prev_sent = 0
dev = False
test = False
sent_len = 0
for sent_id, word, tag in zip(sentence_ids, tokens, tags):
    # determine when to start writing to dev/test files (must be start of a new sent)
    if not dev and index >= dev_start_index and index < test_start_index and sent_id != prev_sent:
        dev = True
    elif not test and index >= test_start_index and sent_id != prev_sent:
        test = True

    if test:
        if sent_id != prev_sent:
            testf.write('\n')
        testf.write(word + ' ' + tag + '\n')
    elif dev:
        if sent_id != prev_sent:
            devf.write('\n')
        devf.write(word + ' ' + tag + '\n')
    else:
        if sent_id != prev_sent:
            trainf.write('\n')
        trainf.write(word + ' ' + tag + '\n')

    if sent_id != prev_sent:
        if sent_len > maxlen:
            maxlen = sent_len
        sent_len = 0
    prev_sent = sent_id
    index += 1
    sent_len += 1

print('maxlen:', maxlen)
trainf.close()
devf.close()
testf.close()
