import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define the network
class News_20_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(News_20_Net, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    # accept input and return an output
    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


def get_batch(df, i, batch_size):
    batches = []
    results = []
    # Split into different batches, get the next batch
    texts = df.title[i * batch_size:i * batch_size + batch_size]
    # get the targets
    categories = df.category[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        # Dimension, 196609
        layer = np.zeros(total_words, dtype=float)

        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)

    # We have 2/4 categories
    if len(c) == 2:
        for category in categories:
            if category == c[0]:
                index_y = 0
            else:
                index_y = 1
            results.append(index_y)
    else:
        for category in categories:
            if category == 'b':
                index_y = 0
            elif category == 't':
                index_y = 1
            elif category == 'e':
                index_y = 2
            else:
                index_y = 3
            results.append(index_y)

    # the training and the targets
    return np.array(batches), np.array(results)


if __name__ == '__main__':
    path = './uci-news-aggregator.csv'
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
    # Generate all possible combinations of two categories from the set b, t, e, m for binary classification,
    # and the entire set, produce training data for each combination and evaluate the model on the test data.
    combinations = [('b', 't'), ('b', 'e'), ('b', 'm'), ('t', 'e'), ('t', 'm'), ('e', 'm'), ('b', 't', 'e', 'm')]
    results = []
    for c in combinations:
        # num_classes is 2 or 4 depending on whether it's binary or not... Categories: b, t, e, m
        if c == ('b', 't', 'e', 'm'):
            num_classes = 4
            train_c, test_c = train.copy(), test.copy()
        else:
            num_classes = 2
            train_c = train[(train['category'] == c[0]) | (train['category'] == c[1])]
            test_c = test[(test['category'] == c[0]) | (test['category'] == c[1])]

        # Getting all the vocabularies and indexing to a unique position
        vocab = Counter()
        # Indexing words from the training data
        for text in train_c.title:
            for word in text.split(' '):
                vocab[word.lower()] += 1

        # Indexing words from the test data
        for text in test_c.title:
            for word in text.split(' '):
                vocab[word.lower()] += 1

        total_words = len(vocab)

        word2index = get_word_2_index(vocab)

        print(len(word2index))
        print(word2index["the"])  # Showing the index of 'the'
        print(total_words)

        # Parameters
        learning_rate = 0.01
        num_epochs = 1
        batch_size = 150
        display_step = 1

        # Network Parameters
        hidden_size = 100  # 1st layer and 2nd layer number of features
        input_size = total_words  # Words in vocab

        news_net = News_20_Net(input_size, hidden_size, num_classes)
        news_net.to(device)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()  # This includes the Softmax loss function
        optimizer = torch.optim.Adam(news_net.parameters(), lr=learning_rate)

        # Train the Model
        for epoch in range(num_epochs):
            # determine the number of min-batches based on the batch size and size of training data
            total_batch = int(len(train_c.title) / batch_size)
            # Loop over all batches
            for i in tqdm(range(total_batch), file=sys.stdout):
                batch_x, batch_y = get_batch(train_c, i, batch_size)
                articles = torch.FloatTensor(batch_x)
                labels = torch.LongTensor(batch_y)
                articles, labels = articles.to(device), labels.to(device)
                # print("articles",articles)
                # print(batch_x, labels)
                # print("size labels",labels.size())

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = news_net(articles)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0 or (i + 1) == total_batch:
                    print(f'\nEpoch: {epoch + 1}/{num_epochs}, Step: {i + 1}/{total_batch}, Loss: {loss.item():.4f}')

        # show the different trained parameters
        for name, param in news_net.named_parameters():
            if param.requires_grad:
                print("Name--->", name, "\nValues--->", param.data)

        # Test the Model
        correct = 0
        total = 0
        total_test_data = len(test_c.category)
        # Modify to use batches for evaluation instead of the entire test set
        total_batch = int(len(test_c.title) / batch_size)
        for i in tqdm(range(total_batch), file=sys.stdout):
            batch_x, batch_y = get_batch(test_c, i, batch_size)
            articles = torch.FloatTensor(batch_x)
            labels = torch.LongTensor(batch_y)
            articles, labels = articles.to(device), labels.to(device)
            outputs = news_net(articles)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print(f'Accuracy of the network on subset {c} of the test articles: {(100 * correct // total)}%')
        results.append([c, (100 * correct // total)])
    for r in results:
        print(f'Accuracy of the network on subset {r[0]} of the test articles: {r[1]}%')
