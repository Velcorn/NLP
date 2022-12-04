import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split

# Prepare training and test dataset: Split the data into training and test set (80% train and 20% test).
# Make sure they are balanced, otherwise if all b files are on training, your model fails to predict t files in test.
path = './uci-news-aggregator.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.lower()
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
# Generate all possible combinations of two categories from the set b, t, e, m,
# produce training data for each combination and evaluate the model on the test data.
combinations = [0]  # [('b', 't'), ('b', 'e'), ('b', 'm'), ('t', 'e'), ('t', 'm'), ('e', 'm')]
for c in combinations:
    """train_c = train[(train['category'] == c[0]) | (train['category'] == c[1])]
    test_c = test[(test['category'] == c[0]) | (test['category'] == c[1])]"""

    train_c, test_c = train.copy(), test.copy()

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


    def get_word_2_index(vocab):
        word2index = {}
        for i, word in enumerate(vocab):
            word2index[word.lower()] = i

        return word2index


    word2index = get_word_2_index(vocab)

    print(len(word2index))
    print(word2index["the"])  # Showing the index of 'the'
    print(total_words)


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
        for category in categories:
            index_y = -1
            if category == 0:
                index_y = 0
            elif category == 1:
                index_y = 1
            elif category == 2:
                index_y = 2
            else:
                index_y = 3
            results.append(index_y)

        # the training and the targets
        return np.array(batches), np.array(results)


    # Parameters
    learning_rate = 0.01   # was .01
    num_epochs = 1  # was 10
    batch_size = 150  # was 150
    display_step = 1  # was 1

    # Network Parameters
    hidden_size = 100  # 1st layer and 2nd layer number of features
    input_size = total_words  # Words in vocab
    num_classes = 4  # Categories: b, t, e, m

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


    news_net = News_20_Net(input_size, hidden_size, num_classes)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # This includes the Softmax loss function
    optimizer = torch.optim.Adam(news_net.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        # determine the number of min-batches based on the batch size and size of training data
        total_batch = int(len(train_c.title) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(train_c, i, batch_size)
            articles = torch.FloatTensor(batch_x)
            labels = torch.LongTensor(batch_y)
            # print("articles",articles)
            # print(batch_x, labels)
            # print("size labels",labels.size())

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = news_net(articles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 16 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1,
                                                                   len(train_c.title) / batch_size, loss.data))

    # show the different trained parameters
    for name, param in news_net.named_parameters():
        if param.requires_grad:
            print("Name--->", name, "\nValues--->", param.data)

    # Test the Model
    correct = 0
    total = 0
    total_test_data = len(test_c.category)
    # get all the test dataset and test them
    batch_x_test, batch_y_test = get_batch(test_c, 0, total_test_data)
    articles = torch.FloatTensor(batch_x_test)
    labels = torch.LongTensor(batch_y_test)
    outputs = news_net(articles)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print(f'Accuracy of the network on the test articles: {(100 * correct // total)} %')
