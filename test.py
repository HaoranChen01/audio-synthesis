import nltk
import simpleaudio
import numpy as np
import re
import calendar
from datetime import datetime
r = re.compile(r'a+')
print(r.match('aaa'))

pattern = r'([a-zA-Z]+)([,]*)([.:?!]*)'
phrase = 'i like apple, but she likes banana. that sounds great! how about you?'
tokens = nltk.tokenize.regexp_tokenize(phrase.lower(), pattern)
print(type(tokens[0][0]))

silence_200 = simpleaudio.Audio(rate=16000)
silence_200.create_tone(0,2000,0)
print(silence_200.data.shape[0])

print(np.linspace(1,0,10)[-10:0])
print(np.array([[]])[:-5])

time = datetime.strptime('8/12/89','%d/%m/%y')
print(time)
print(time.strftime('%B'))
print(time.strftime('%y'))
print(nltk.corpus.cmudict.dict()['three'][0])

r2 = re.compile(r'((\d+)\/(\d+)\/((19\d\d)|(\d\d)))|((\d+)\/(\d+))')
a = '20/5/20'
b = '23/3'
c = r2.match(a).group(1)
d = r2.match(b).group(7)
print(c[:-5],d)