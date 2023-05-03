"""
 * Project TF IDF Vectorizer
 *
 * @author      Moin Khan
 * @copyright   Moin Khan
 *
 * @link https://mo.inkhan.dev
 *
 */
 """

from tf_idf_vectorizer import get_tf_idf_with_terms

documents = [
    "Lions are large carnivorous mammals that belong to the Felidae family. They inhabit grasslands and savannas in Africa and Asia.",
    "Elephants are the largest land mammals on Earth. They are herbivores and have long trunks, large ears, and strong tusks.",
    "Dolphins are highly intelligent marine mammals known for their agility, playful behavior, and complex communication.",
    "Giraffes are the tallest mammals on Earth, with long necks and legs that allow them to feed on leaves high in trees.",
]

terms, tf_idf_vectors = get_tf_idf_with_terms(documents)

print("Terms:", terms)
for i, doc in enumerate(tf_idf_vectors):
    print(f"Document {i+1}: {doc}")
