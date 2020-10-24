# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:15:50 2019

@author: Akshay
"""

from os import listdir
import string

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '\\' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story':story, 'highlights':highlights})
    return stories
# identifies if the given string is a number or not 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('PUBLISHED:')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        temp = []                           
        for word in line:
            if word.isalpha():
                temp.append(word)   
            if is_number(word):
                for num in list(word):
                    temp.append(num)
        line = temp
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned

# load stories
directory = r'dailymail\stories'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

# clean stories
for example in stories:
    example['story'] = clean_lines(example['story'].split('\n'))
    example['highlights'] = clean_lines(example['highlights'])
for example in stories:
    example['story'] = ".".join(example['story'])
    #example['highlights'] = clean_lines(example['highlights'])

# save to file
from pickle import dump
dump(stories, open('dailymail_dataset.pkl', 'wb'))

# load from file
from pickle import load
DM = load(open('dailymail_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(DM))

from pickle import load
CNN = load(open('cnn_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(CNN))
count=0
highlights =''
for i in CNN:
    count+=1
    print(count)
    g=''
    for j in i['highlights']:
        g+=j+'   '
    highlights+=g+'\n'

count=0
story =''
for i in CNN:
    count+=1
    print(count) 
    story+=i['story']+'\n'

with open('CNN_highlights.txt','w',encoding='utf-8') as a:
    a.write(highlights)
with open('CNN_story.txt','w',encoding='utf-8') as a:
    a.write(story)

count=0
highlights =''
for i in DM:
    count+=1
    print(count)
    g=''
    for j in i['highlights']:
        g+=j+'   '
    highlights+=g+'\n'

count=0
story =''
for i in DM:
    count+=1
    print(count) 
    story+=i['story']+'\n'

with open('DM_highlights.txt','w',encoding='utf-8') as a:
    a.write(highlights)
with open('DM_story.txt','w',encoding='utf-8') as a:
    a.write(story)