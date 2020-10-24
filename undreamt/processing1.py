# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:54:39 2019

@author: Akshay
"""

a = open('hindencorp05.plaintext',encoding='utf-8')
data = a.readlines()
import string
for i in range(len(data)):
    data[i] = data[i].strip().split('\t')[3:]

english=[]
hindi=[]
for i in data:
    english.append(i[0])
    hindi.append(i[1])
    


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        """index = line.find('PUBLISHED:')
        if index > -1:
            line = line[index+len('(CNN)'):]"""
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
    cleaned = [c for c in cleaned] #if len(c) > 0]
    return cleaned
from HindiTokenizer import Tokenizer
english = clean_lines(english)
hindi = clean_lines(hindi)

table = str.maketrans('', '', string.punctuation)
ind=[]
for i in range(len(english)):
    if len(english[i])==0:
        ind.append(i)

hindi_text=''
for sentence in hindi:
    hindi_text+= sentence +'\n'
with open("hindi_exp_text.txt","w",encoding="utf-8") as a:
    a.write(hindi_text)
english_text=''
for sentence in english:
    english_text+= sentence +'\n'
with open("english_exp_text.txt","w",encoding="utf-8") as a:
    a.write(english_text)
#experiments
t = Tokenizer()
data_hindi=t.read_from_file('hindi_exp_text.txt')
k='शराबी'
freq_dict = t.generate_freq_dict()


# vocab
import collections
temp=english_text.split()
vocab_eng = collections.Counter(temp)
vocab_eng = zip(vocab_eng.keys(),vocab_eng.values())
vocab_eng = list(vocab_eng)
vocab_eng = sorted(vocab_eng,key=lambda x : x[1],reverse=True)
vocab_eng = vocab_eng[:30000]
vocab_eng_text = ''
for word,count in vocab_eng:
    vocab_eng_text+=word+"\n"

temp=hindi_text.split()
vocab_hin = collections.Counter(temp)
vocab_hin = zip(vocab_hin.keys(),vocab_hin.values())
vocab_hin = list(vocab_hin)
vocab_hin = sorted(vocab_hin,key=lambda x : x[1],reverse=True)
vocab_hin = vocab_hin[:30000]
vocab_hin_text = ''
for word,count in vocab_hin:
    vocab_hin_text+=word+"\n"