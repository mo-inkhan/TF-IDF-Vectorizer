# TF-IDF Vectorizer

This Python-based TF-IDF Vectorizer is a simple implementation of the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm. It takes a list of text documents as input and calculates a TF-IDF matrix, which can be used for various Natural Language Processing (NLP) and Machine Learning tasks. Compute the TF-IDF matrix from a collection of documents to measure the importance of words for text analysis and information retrieval tasks.

## Features

- Tokenization of input documents
- Calculation of Term Frequency (TF) for each term in each document
- Calculation of Inverse Document Frequency (IDF) for each term in the corpus
- Calculation of TF-IDF matrix from input documents

## Usage

1. Import the `get_tf_idf_with_terms` function from the `tf_idf_vectorizer.py` module:

```python
from tf_idf_vectorizer import get_tf_idf_with_terms
```

2. Pass a list of documents (strings) to the `get_tf_idf_with_terms` function:

```python
documents = [
    "A young wizard discovers his magical heritage and begins his studies at a prestigious school for wizards.",
    "A group of astronauts embark on a dangerous mission to save Earth by entering a wormhole in search of a new habitable planet.",
    "In a post-apocalyptic world, a father and son journey through a desolate landscape while trying to survive and find hope for humanity.",
    "An aspiring musician enters a magical world to find his true passion and learn what it means to live a fulfilled life.",
]
```

3. The `get_tf_idf_with_terms` function returns a tuple containing the unique terms (column keys) and the calculated TF-IDF matrix as a list of lists:

```python
unique_terms, tf_idf_matrix = get_tf_idf_with_terms(documents)

print("Unique terms:", unique_terms)
for i, doc in enumerate(tf_idf_matrix):
    print(f"Document {i+1}: {doc}")
```


## Example
Check `example.py` for sample use case.


## Limitations
This implementation is intended for educational purposes and may not be as efficient or robust as more advanced libraries. It does not handle stopwords, punctuation, or stemming, which may be needed in a more advanced implementation.


## Contributing
All contributions are welcome. Please create an issue first for any feature request or bug. Then fork the repository, create a branch and make any changes to fix the bug or add the feature and create a pull request. That's it!
Thanks!


## License
**TF-IDF Vectorizer** is released under the MIT License.
Check out the full license [here](LICENSE).
