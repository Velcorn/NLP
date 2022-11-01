import nltk
# nltk.download('all')
from nltk.book import *

# The code speaks for itself - Jan Willruth 2022

# 1.
with open('data/news/news.txt', 'r') as f:
    news = f.read().split()
    num_tokens = len(news)
    num_types = len(set(news))
    print('Number of tokens: ', num_tokens)
    print('Number of types: ', num_types)

# 2.
with open('data/news/news1.txt', 'r') as f:
    news1 = f.read().split()
    print(f'Intersection: {set(news) & set(news1)}')

# 3.
books = [*text1.tokens, *text2.tokens, *text3.tokens, *text4.tokens, *text5.tokens, *text6.tokens, *text7.tokens,
         *text8.tokens, *text9.tokens]
for l in [chr(x) for x in range(ord('a'), ord('z')+1)]:
    try:
        print(f'Longest word starting with {l}: {max([w for w in books if w.startswith(l)], key=len)}')
    except ValueError:
        print(f'Longest word starting with {l}: ')

# 4.
starting_letters = [w[0] for w in books if len(w) > 0]
ending_letters = [w[-1] for w in books if len(w) > 0]
print(f'Most frequent starting letter: {max(set(starting_letters), key=starting_letters.count)}')
print(f'Most frequent ending letter: {max(set(ending_letters), key=ending_letters.count)}')

# 5.
