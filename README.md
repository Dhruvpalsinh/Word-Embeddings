# Word Embeddings

This project explores different ways to represent text numerically in NLP and machine learning. The notebook demonstrates core text representation techniques such as Bag of Words (BoW), N-grams, TF-IDF, and Word2Vec.

## Project Overview

The notebook covers the following concepts:

- Corpus, vocabulary, document, and word
- Bag of Words representation using `CountVectorizer`
- Vocabulary mapping and sparse matrix generation
- N-grams for capturing word sequences
- TF-IDF for weighting important terms
- Word2Vec preprocessing with `gensim`
- Sentence tokenization and text cleaning

## Topics Covered

### 1. Bag of Words

The notebook explains how text can be converted into numerical vectors where each unique word becomes a feature. It shows how the vocabulary is built and how each sentence is transformed into a count vector.

Example:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

bow = cv.fit_transform(df['text'])
print(cv.vocabulary_)
print(bow.toarray())
```

### 2. N-grams

The notebook demonstrates how word pairs (bi-grams) can capture local word context better than single words alone.

```python
cv = CountVectorizer(ngram_range=(2, 2))
```

### 3. TF-IDF

The notebook shows how TF-IDF helps assign more weight to important terms while reducing the impact of frequent but less informative words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
arr = tfidf.fit_transform(df['text']).toarray()
```

### 4. Word2Vec

The notebook includes a Word2Vec demo using `gensim` and NLP preprocessing utilities. It demonstrates:

- tokenization with `sent_tokenize`
- text cleaning with `simple_preprocess`
- reading documents from a `data` folder
- building tokenized sentence lists for model training

```python
from nltk import sent_tokenize
from gensim.utils import simple_preprocess

story.append(simple_preprocess(sent))
```

## Requirements

Install the required libraries:

```bash
pip install numpy pandas scikit-learn nltk gensim
```

## Notebook File

Open and run:

```bash
TEXT_REPRESENTATION_WORD_EMBEDDINGS.ipynb
```

This notebook is designed to run in Jupyter Notebook or Google Colab.

## Notes

- The Word2Vec section expects a `data/` directory containing text files.
- In the notebook, this directory is used to build a corpus from multiple text documents.
- If the `data` folder is missing, the notebook may raise a `FileNotFoundError` until the corpus files are added.

## Example Dataset Used

The notebook creates a small custom dataset for demonstration:

```python
import pandas as pd

pd.DataFrame({
    'text': [
        'people watch campusx',
        'campusx watch campusx',
        'people write comment',
        'campusx write comment'
    ],
    'output': [1, 1, 0, 0]
})
```

## Learning Outcomes

After working through this notebook, you will understand:

- how text is transformed into vectors
- how word order and context can be handled with n-grams
- how TF-IDF improves feature weighting
- how Word2Vec creates dense word embeddings

## Author

Dhruvpalsinh

## License

This project is intended for educational and learning purposes.
