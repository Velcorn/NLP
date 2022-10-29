import csv
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from glob import glob
from germalemma import GermaLemma
from pattern.de import parse, split

# 1.1
# List all the files in the subdirectories of the data folder that you have downloaded from Moodle.
files = glob('data/*/*')
for file in files:
    print(file)

# 1.2
# Merge all the news content into a single file and save into a different file named allnews.txt using the ISO-8859-1
# encoding.
with open('data/news/allnews.txt', 'w', encoding='ISO-8859-1') as f:
    for file in glob('data/news/*'):
        with open(file, 'r', encoding='ISO-8859-1') as g:
            f.write(g.read())

# 1.3
# Read the country name and capital city from this (https://geographyfieldwork.com/WorldCapitalCities.htm) page,
# which lists the world capital cities with their country. Save the result as a comma separated value (csv) file format.
url = 'https://geographyfieldwork.com/WorldCapitalCities.htm'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
country_capital = {'Country': 'Capital City'}
for i, p in enumerate(soup.find_all('td')):
    if i % 2 == 0:
        text = p.text
        if text[0].isalpha():
            country_capital[text] = soup.find_all('td')[i + 1].text
with open('data/country_capital.csv', 'w') as f:
    writer = csv.writer(f)
    for key, value in country_capital.items():
        writer.writerow([key, value])


# 2.1
text = "Fruits like apple, orange, and mango are healthy. But they are expensive, i.e, Mr. Bean can't afford them! " \
       "One can order some online from www.rewe.de. Prof. Karl, Dep. of Plant Science. " \
       "Email: karl@plant.science.de. Regards!"
# Make pattern that splits sentences properly
pattern = r'(?<=[.?!])\s+(?=[A-Z])'
print(re.split(pattern, text))

# 2.2
# Modify pattern for ideal tokenization
text = "\"I said, 'what're you? Crazy?'\" said Sandowsky. \"I can't afford to do that.\""
pattern = r'(\w+|\$[\d\.]+|\S+)'
print(re.split(pattern, text))

# 3
sentence2 = """Die Brände in Brasilien setzen erhebliche Mengen an klimaschädlichen Treibhausgasen frei. 
Die Nasa hat nun simuliert, wie sich Kohlenmonoxid über Südamerika ausbreitet.
Am Boden schadet das Gas der Gesundheit erheblich."""
de_lemma = GermaLemma()
# POS tagger
poses = parse(sentence2)
for sentence in split(poses):
    print("===")
    for token in sentence:
        # print(token, token.pos)
        # TODO: Here map the POS tag to V, N, ADJ, or ADV as MAPD_POS. MAPD_POS = ???
        # If the POS tag is not in V, N, ADJ, or ADV, no need to lemmatize
        if token.pos in ['NN', 'NNS']:
            mapped_pos = 'N'
        elif token.pos in ['VB']:
            mapped_pos = 'V'
        elif token.pos in ['JJ']:
            mapped_pos = 'ADJ'
        elif token.pos in []:
            mapped_pos = 'ADV'
        else:
            mapped_pos = token
        # Print the lemma here
        if mapped_pos in ['N', 'V', 'ADJ', 'ADV']:
            print(de_lemma.find_lemma(token, mapped_pos))
        else:
            print(token)
    print("===")
