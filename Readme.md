# Document Similarity Matcher

## Introduction

This project is a document similarity matcher that uses the cosine and jaquard similarity to compare the similarity between two documents. The project is implemented in Python.

## Intuition

The project uses cosine similarity and Jaccard similarity to compare documents. Cosine similarity measures how closely two documents' word vectors align, indicating similarity based on word frequency. Jaccard similarity assesses similarity by comparing the overlap of words in each document. The process involves extracting text from PDFs, preprocessing it by means of the LLM meta-llama/llama-3-8b-instruct, and then calculating these similarities to evaluate how closely documents match in content and word usage.

## Methodology

- Extract Text: Use PyMuPDF and pdfplumber to extract text and tables from the PDF.
- Preprocess Text: Clean and prepare the text data for analysis.
- Extract Fields with AI: Use the OpenAI model to extract structured information from the text.
- Compute Similarities:
  Use scikit-learn to calculate cosine similarity based on TF-IDF vectors.
  Compute Jaccard similarity by comparing the sets of words in the documents.
- Evaluate Similarity: Combine the similarity scores to determine how closely documents match.

## Tech Stack

- Python
- Numpy
- fitz
- pdfplumber
- cosine_similarity
- jacquard_similarity
- argparse
- os
- OpenRouter API
- OpenAI

## Installation

1. Clone the repository
2. Set up the database in the following format

```
Dataset/
    train/
        file.pdf
        file2.pdf
    test/
        file.pdf
        file2.pdf
```

4. Run the following command to install the required packages

```
pip install -r requirements.txt
```

5. Run the following command to run the project

- For single file comparison

```
python index.py --train_dir "<train_path>" --file "<file_path>"
```

- For multiple file comparison

```
python index.py --train_dir "<train_path>" --test_dir "<test_path>" --multiple True
```
