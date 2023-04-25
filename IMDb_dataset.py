# from tqdm import tqdm
# import random
# import numpy as np
# import os
#     # Write a tokenizer.
# def tokenizer(sentence):
#     tokens = sentence.split()
#     tokens = [token.lower() for token in tokens]
#     return tokens

# def read_imdb(folder, data_root="./data/aclImdb"): 
#         """
#         data: list of [string, label]
#         """
#         data = []
#         for label in ['pos', 'neg']:
#             folder_name = os.path.join(data_root, folder, label)
#             for file in tqdm(os.listdir(folder_name)):
#                 with open(os.path.join(folder_name, file), 'rb') as f:
#                     review = f.read().decode('utf-8').replace('\n', '').lower()
#                     data.append([review, 1 if label == 'pos' else 0])
#         random.shuffle(data)
#         return data

# def get_data(train_test):
    
#     # train_data, test_data = read_imdb('train'), read_imdb('test')
#     test_data = read_imdb(train_test)

#     # Create a vocabulary.
#     vocabulary = set()
#     for sentence,_ in test_data:
#         tokens = tokenizer(sentence)
#         for token in tokens:
#             vocabulary.add(token)

#     vocabulary = sorted(list(vocabulary))

#     # Embed the tokens.
#     tokenized_sentences = []
#     for sentence,_ in test_data:
#         tokens = tokenizer(sentence)
#         tokenized_sentence = np.zeros(len(tokens))
#         for i, token in enumerate(tokens):
#             tokenized_sentence[i] = vocabulary.index(token)
#         tokenized_sentences.append(tokenized_sentence)

#     # Pad the tokens.
#     max_sentence_length = 500
#     for i, tokenized_sentence in enumerate(tokenized_sentences):
#         if len(tokenized_sentence) < max_sentence_length:
#             tokenized_sentences[i] = np.pad(tokenized_sentence, (0, max_sentence_length - len(tokenized_sentence)),'constant', constant_values=(0, 0))
#         tokenized_sentences[i] = tokenized_sentences[i][:max_sentence_length]

#     # Return the embedding array.
#     return tokenized_sentences


# if __name__ == "__main__":
#     print(get_data("train").shape)



from tqdm import tqdm
import random
import numpy as np
import os
from collections import Counter
import re
def tokenizer(sentence):
    tokens = sentence.split()
    tokens = [token.lower() for token in tokens]
    return tokens


def read_imdb(folder, data_root="./data/aclImdb"):
    """
    data: list of [string, label]
    """
    data = []
    for label in ["pos", "neg"]:
        folder_name = os.path.join(data_root, folder, label)
        folders = os.listdir(folder_name)
        # for file in tqdm(os.listdir(folder_name)):
        for i in range(10):
            file = folders[i]
            with open(os.path.join(folder_name, file), "rb") as f:
                review = f.read().decode("utf-8").replace("\n", "").lower()
                review = re.sub(r"\([^()]*\)", "", review) # remove ([text])
                review = review.replace('-', '') # remove '-'
                review = re.sub('([.,;!?()\"])', r' \1 ', review) # keep punctuation
                data.append([review, 1 if label == "pos" else 0])
    random.shuffle(data)
    return data



def get_data(train_test):
    """
    Returns the train or test data as a NumPy array of shape (num_examples, max_sentence_length).
    """
    data = read_imdb(train_test)

    # Create a vocabulary.
    # vocabulary = set()
    # tokens = []
    # for sentence, _ in data:
    #     for token in tokenizer(sentence):
    #         tokens.append(token)
    # token_frequencies = Counter(tokens)

    # filtered_tokens = [token for token, frequency in token_frequencies.items() if frequency >= 5]
    # vocabulary = sorted(list(filtered_tokens))

    # Create a vocabulary.
    vocabulary = set()
    for sentence,_ in data:
        tokens = tokenizer(sentence)
        for token in tokens:
            vocabulary.add(token)

    vocabulary = sorted(list(vocabulary))


    num_tokens = len(vocabulary)
    print(num_tokens)
    # Embed the tokens.
    labels = []
    max_sentence_length = 500
    tokenized_sentences = np.zeros((len(data), max_sentence_length), dtype=int)
    for i, (sentence, label) in enumerate(data):
        labels.append(label)
        tokens = tokenizer(sentence)
        for j, token in enumerate(tokens):
            if j >= max_sentence_length:
                break
            tokenized_sentences[i, j] = vocabulary.index(token)

    # Pad the tokens.
    for i in range(len(tokenized_sentences)):
        tokenized_sentences[i] = np.pad(tokenized_sentences[i], (0, max_sentence_length - len(tokenized_sentences[i])), "constant", constant_values=(0, 0))

    return tokenized_sentences,np.array(labels),num_tokens


if __name__ == "__main__":
    print(get_data("train")[0].shape,get_data("train")[1].size)
