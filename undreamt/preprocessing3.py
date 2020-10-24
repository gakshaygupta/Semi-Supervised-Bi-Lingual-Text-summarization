# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:07:21 2019

@author: Akshay
"""


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
def alpha(line):
    temp=[]
    for word in line.split():
        if is_number(word):
            for num in list(word):
                temp.append(num)
        else:
            temp.append(word)
    return " ".join(temp)
def clean_lines(line):
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    
    # strip source cnn office if it exists
    """ index = line.find('PUBLISHED:')
    if index > -1:
        line = line[index+len('(CNN)'):]"""
    # tokenize on white space
    line = line.split()
    # convert to lower case
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [w.translate(table) for w in line]
    
    temp = []                           
    for word in line:
        if word.isalpha():
            temp.append(word)   
        if is_number(word):
            for num in list(word):
                temp.append(num)
    line = temp
    # store as string
    
    # remove empty strings
    #cleaned = [c for c in cleaned if len(c) > 0]
    return ' '.join(line) + '\n'
def clean_lines_hindi(line):
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
    
    # strip source cnn office if it exists
    """ index = line.find('PUBLISHED:')
    if index > -1:
        line = line[index+len('(CNN)'):]"""
    # tokenize on white space
    line = line.split()
    # convert to lower case
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [w.translate(table) for w in line]
    
    temp = []                           
    for word in line:
        if is_number(word):
            for num in list(word):
                temp.append(num)
        else:
            temp.append(word) 
    line = temp
    # store as string
    
    # remove empty strings
    #cleaned = [c for c in cleaned if len(c) > 0]
    return ' '.join(line)
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

def hindi_clean(data):
    input_text=data
    remove_nuktas=False
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer("hi",remove_nuktas)
    output_text=normalizer.normalize(input_text)
    output_text=clean_lines_hindi(output_text)
    output_text=alpha(output_text)
    return output_text+'\n'
a = open("IITB.en-hi.en","r",encoding="utf-8")
b = open("IITB.en-hi.hi",'r',encoding = "utf-8")

#data = a.readline()

count = 0
data_hindi = ''
data_english = ''
while True:
    try:
        sentence_hindi = b.readline()
        sentence_english = a.readline()
        if len(sentence_hindi)==0 and len(sentence_english)==0:
            break
        sentence_hindi = hindi_clean(sentence_hindi)
        sentence_english = clean_lines(sentence_english)
        if 10 <len(sentence_english.strip().split()) <=50 and 10 <len(sentence_hindi.strip().split()) <=50:
            data_hindi = data_hindi + sentence_hindi
            data_english = data_english + sentence_english
            
            count+=1
            if count%1000==0:
                print(count)
    except UnicodeDecodeError:
        pass
print(count)
a.close()
b.close
with open("english_mono_50000_news_crawl.txt","w",encoding="utf-8") as english:
    english.write(data)
    
counter=0
while a.readable():
    i=a.readline()
    counter+=1
    if counter%10000==0:
        print(counter)

import collections
temp=data.split()
vocab_eng = collections.Counter(temp)
vocab_eng = zip(vocab_eng.keys(),vocab_eng.values())
vocab_eng = list(vocab_eng)
vocab_eng = sorted(vocab_eng,key=lambda x : x[1],reverse=True)
vocab_eng = vocab_eng[:50000]
vocab_eng_text = ''
for word,count in vocab_eng:
    vocab_eng_text+=word+"\n"
with open("english_vocab.txt","w",encoding="utf-8") as a:
    a.write(vocab_eng_text)