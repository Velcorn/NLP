from glob import glob
from nltk import tokenize

input_path = 'input'
texts = [open(f).read() for f in glob(f'{input_path}/*.txt')]
sentences = [tokenize.sent_tokenize(t) for t in texts]
hyponym_hypernym = {}
for s in sentences:
    pass
