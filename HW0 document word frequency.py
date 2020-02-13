from os import listdir
from collections import Counter
from itertools import combinations
from re import sub

def most_frequent(word_set_length = 3):
    """The most frequent "wordset" is defined as the "wordset" which is a subset of the text of the most documents"""
    word_frequencies = Counter()
    for file in listdir("docs/"):
        word_frequencies += Counter(combinations(words(file), r=word_set_length))

    return word_frequencies.most_common(1)


def words(file_name):
    """returns set of words in file"""
    set_of_words = set()
    with open(f"docs/{file_name}", "r", encoding="UTF-8") as file:
        for line in file.readlines():
            set_of_words |= {sub(r'\W+', '', word) for word in line.split()}
    return set_of_words


for x in range(1, 4):
    print(most_frequent(x))