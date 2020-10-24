# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:39:54 2019

@author: Akshay
"""
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def alpha(line):
    temp=[]
    for word in line.split():
        if is_number(word):
            for num in list(word):
                temp.append(num)
        else:
            temp.append(word)
    return " ".join(temp)
def clean_text(text):
    '''not working'''
    
    #text=re.sub(r'(\d+)',r'',text)
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",'')
    text=text.replace(u"@",'')
    text=text.replace(u"%",'')
    text=text.replace(u"$",'')
    text=text.replace(u"*",'')
    text=text.replace(u"-",'')
    text=text.replace(u"&",'')
    text=text.replace(u"!",'')
    text=text.replace(u"^",'')
    text=text.replace(u"#",'')
    text=text.replace(u"*",'')
    text=text.replace(u"=",'')
    text=text.replace(u"<",'')
    text=text.replace(u">",'')
    text=text.replace(u"_",'')
    text=text.replace(u"*",'')
    
    #!"#$%&'()*+,-.:;<=>?@[]^_`{|}~

    return text

def clean_lines(line):
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
def hindi_clean(data):
    input_text=data
    remove_nuktas=False
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer("hi",remove_nuktas)
    output_text=normalizer.normalize(input_text)
    output_text=clean_lines(output_text)
    output_text=alpha(output_text)
    return output_text+'\n'


a = open("monolingual.hi","r",encoding="utf-8")
#data = a.readline()

count = 0
data = ''
while count != 500000:
    try:
        sentence = a.readline()
        sentence = hindi_clean(sentence)
        if 10 <len(sentence.strip().split()) <=50:
            data = data + sentence
            count+=1
            if count%1000==0:
                print(count)
    except UnicodeDecodeError:
        pass
a.close()
with open("hindi_mono_50000_iitB.txt","w",encoding="utf-8") as hindi:
    hindi.write(data)
with open("hindi_mono_50000_iitB.txt","r",encoding="utf-8") as hindi:
    data = hindi.readlines()
import re


INDIC_NLP_LIB_HOME=r"C:\Users\Akshay\Documents\GitHub\indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=r"C:\Users\Akshay\Documents\GitHub\indic_nlp_resources"
import sys
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()

# formating


from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

input_text=data[2]
remove_nuktas=True
factory=IndicNormalizerFactory()
normalizer=factory.get_normalizer("hi",remove_nuktas)
output_text=normalizer.normalize(input_text)
print(input_text)
print(output_text)
print('Length before normalization: {}'.format(len(input_text)))
print('Length after normalization: {}'.format(len(output_text)))

line = data[1000]
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
with open("hindi_vocab.txt","w",encoding="utf-8") as a:
    a.write(vocab_eng_text)
    
    
    


#def clean_lines(line):
#    # prepare a translation table to remove punctuation
#    table = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
#    
#    # strip source cnn office if it exists
#    """ index = line.find('PUBLISHED:')
#    if index > -1:
#        line = line[index+len('(CNN)'):]"""
#    # tokenize on white space
#    line = line.split()
#    # convert to lower case
#    line = [word.lower() for word in line]
#    # remove punctuation from each token
#    line = [w.translate(table) for w in line]
#    
#    temp = []                           
#    for word in line:
#        if is_number(word):
#            for num in list(word):
#                temp.append(num)
#        else:
#            temp.append(word) 
#    line = temp
#    # store as string
#    
#    # remove empty strings
#    #cleaned = [c for c in cleaned if len(c) > 0]
#    return ' '.join(line) + '\n'