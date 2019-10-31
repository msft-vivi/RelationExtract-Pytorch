"""Read and save the semeval dataset for our model"""

import os
import re

pattern_repl = re.compile('(<e1>)|(</e1>)|(<e2>)|(</e2>)|(\'s)')
pattern_e1 = re.compile('<e1>(.*)</e1>')
pattern_e2 = re.compile('<e2>(.*)</e2>')
pattern_symbol = re.compile('^[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]|[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]$')


def load_dataset(path_dataset):
    """Load dataset into memory from text file"""
    dataset = []
    with open(path_dataset) as f:
        piece = list()  # a piece of data
        for line in f:
            line = line.strip()
            if line:
                piece.append(line)
            elif piece:
                sentence = piece[0].split('\t')[1].strip('"')
                e1 = delete_symbol(pattern_e1.findall(sentence)[0])
                e2 = delete_symbol(pattern_e2.findall(sentence)[0])
                new_sentence = list()
                for word in pattern_repl.sub('', sentence).split(' '):
                    new_word = delete_symbol(word)
                    if new_word:
                        new_sentence.append(new_word)

                relation = piece[1]
                dataset.append(((e1, e2, ' '.join(new_sentence)), relation))
                piece = list()
    return dataset

def delete_symbol(text):
    if pattern_symbol.search(text):
        return pattern_symbol.sub('', text)
    return text

def save_dataset(dataset, save_dir):
    """Write `sentences.txt` and `labels.txt` files in save_dir from dataset"""
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    """words : ('garbage bag', 'clothes', 'I have a large garbage bag full of clothes for a teen or preteen girl')"""
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, \
        open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
        for words, labels in dataset:
            file_sentences.write('{}\n'.format('\t'.join(words)))
            file_labels.write('{}\n'.format(labels))
    print("- done.")


if __name__ == '__main__':
    path_train = 'data/SemEval2010_task8/TRAIN_FILE.TXT'
    path_test = 'data/SemEval2010_task8/TEST_FILE.TXT'
    msg = "{} or {} file not found. Make sure you have downloaded the right dataset".format(path_train, path_test)
    assert os.path.isfile(path_train) and os.path.isfile(path_test), msg

    # load the dataset into memory
    print("Loading SemEval2010_task8 dataset into memory...")
    train_dataset = load_dataset(path_train)
    test_dataset = load_dataset(path_test)
    print("- done.")

    # save the dataset to text file
    save_dataset(train_dataset, 'data/SemEval2010_task8/train')
    save_dataset(test_dataset, 'data/SemEval2010_task8/test')