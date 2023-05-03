"""
 * Project TF-IDF Vectorizer
 *
 * @author      Moin Khan
 * @copyright   Moin Khan
 *
 * @link https://mo.inkhan.dev
 *
 */
 """

import math
from collections import defaultdict
from typing import List


def __tokenize(document: str) -> list:
    """
    Tokenize the document

    :param document: string - The document to be tokenized
    :return list - The list of tokens
    """

    return document.lower().split(" ")


def __tf_evaluator(document: list) -> dict:
    """
    Term Frequency

    :param document: list - List containing document tokens
    :return dict - Dictionary of term frequency
    """

    tf = defaultdict(lambda: 0)

    for term in document:
        tf[term] += 1 / len(document)

    return tf


def __idf_evaluator(documents: List[list]) -> dict:
    """
    Inverse Document Frequency

    :param documents: List[list] - List of document tokens
    :return dict - Dictionary of IDFs
    """

    idf = defaultdict(lambda: 0)

    for doc in documents:
        doc = set(doc)

        for term in doc:
            idf[term] += 1

    for term, value in idf.items():
        idf[term] = math.log(len(documents) / value)

    return idf


def get_tf_idf_with_terms(documents: List[str]) -> list:
    """
    Get TF IDF of a given document list

    :param documents: List[str] - List of documents
    :return list - Matrix Representing TF IDF
    """

    tf_idf_matrix = []
    tokenized_documents = [__tokenize(doc) for doc in documents]
    idf = __idf_evaluator(tokenized_documents)
    term_keys = sorted(idf.keys())

    for doc in tokenized_documents:
        tf = __tf_evaluator(doc)
        tf_idf_row = []

        for term in term_keys:
            tf_idf_row.append(tf[term] * idf[term])

        tf_idf_matrix.append(tf_idf_row)

    return term_keys, tf_idf_matrix
