import fitz
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
from openai import OpenAI


# Extract text using PyMuPDF
def extract_text_with_pymupdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text() + "\n"
    return text


# Extract tables using pdfplumber
def extract_tables_with_pdfplumber(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:  # Ensure table is not empty
                    tables.append(table)
    return tables


# Combine text and tables
def combine_text_and_tables(text, tables):
    combined_data = text
    combined_data += "\n\nTables:\n"
    for table in tables:
        for row in table:
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            combined_data += "\t".join(cleaned_row) + "\n"
        combined_data += "\n"
    return combined_data


# Extract fields using OpenAI model
def extract_fields_with_openai(data):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    completion = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": "I need you to extract the following data from the given message. The attributes are Invoice Number, Invoice Date, Due Date, Purchase Order Number, Vendor Name, Vendor Address, Vendor Contact Information, Customer Name, Customer Address, Customer Contact Information, Description, Quantity, Unit Price, Total Price, Item Code/SKU, Subtotal, Taxes, Discounts, Total Amount Due, Payment Terms, Bank Account Details, Payment Instructions, Terms and Conditions, Notes, Attachments, Currency, Invoice Status, Reference Numbers. Please extract the data from the given message. The data is as follows: "
                + data,
            },
        ],
    )
    return completion.choices[0].message.content


# Read all data from PDFs in a directory and extract fields using OpenAI model
def read_all_pdfs_in_directory(directory):
    pdf_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_with_pymupdf(pdf_path)
            tables = extract_tables_with_pdfplumber(pdf_path)
            combined_data = combine_text_and_tables(text, tables)
            extracted_fields = extract_fields_with_openai(combined_data)
            pdf_data[filename] = extracted_fields
    return pdf_data


# Calculate Jaccard similarity
def jaccard_similarity(doc1, doc2):
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


# Calculate Cosine similarity
def cosine_similarity_documents(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


# Find the most similar document
def find_most_similar_document(train_data, test_doc):
    max_similarity = 0
    most_similar_doc = ""
    for doc_name, doc_data in train_data.items():
        jaccard = jaccard_similarity(test_doc, doc_data)
        cosine = cosine_similarity_documents(test_doc, doc_data)
        avg_similarity = (jaccard + cosine) / 2
        if avg_similarity > max_similarity:
            max_similarity = avg_similarity
            most_similar_doc = doc_name
    return most_similar_doc, max_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Document Similarity Checker")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="E:/document_similarity/Dataset/train",
        help="Path to the training directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="E:/document_similarity/Dataset/test",
        help="Path to the test directory",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single PDF file to check against the training dataset",
    )

    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir

    # Read all PDFs in the training directory
    train_data = read_all_pdfs_in_directory(train_dir)

    if args.file:
        # Process a single file
        test_doc_path = args.file
        test_doc_name = os.path.basename(test_doc_path)
        text = extract_text_with_pymupdf(test_doc_path)
        tables = extract_tables_with_pdfplumber(test_doc_path)
        combined_data = combine_text_and_tables(text, tables)
        test_doc_data = extract_fields_with_openai(combined_data)
        most_similar_doc, max_similarity = find_most_similar_document(
            train_data, test_doc_data
        )
        print(f"Test Document: {test_doc_name}")
        print(f"Most Similar Document: {most_similar_doc}")
        print(f"Similarity Score: {max_similarity}\n")
    else:
        # Process all files in the test directory
        test_data = read_all_pdfs_in_directory(test_dir)
        for test_doc_name, test_doc_data in test_data.items():
            most_similar_doc, max_similarity = find_most_similar_document(
                train_data, test_doc_data
            )
            print(f"Test Document: {test_doc_name}")
            print(f"Most Similar Document: {most_similar_doc}")
            print(f"Similarity Score: {max_similarity}\n")
