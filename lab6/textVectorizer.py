import os
import numpy as np
import argparse
from collections import defaultdict, Counter
import csv
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('punkt')  # Ensure tokenization tools are available
nltk.download('stopwords')  # Ensure stopwords are available

class textVectorizer:
    def __init__(self, base_dir, remove_stopwords=False, apply_stemming=False):
        self.base_dir = base_dir
        self.stemmer = PorterStemmer() if apply_stemming else None
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
        self.documents, self.ground_truth = self.load_documents()
        self.tfidf_matrix = None
        self.vocab = None

        self.okapi_lookup = None
        self.doc_term_counts = None
        self.term_doc_freq = None

    def preprocess_text(self, text):
        '''
        Preprocess the given text document by:
            - Removing punctuation.
            - Converting text to lowercase.
            - Removing stopwords (if specified).
            - Applying stemming (if specified)

        Returns a list of each word in the document
        '''
        words = text.translate(str.maketrans('', '', string.punctuation)).lower().split()
        words = [word for word in words if word not in self.stopwords]

        if self.stemmer:
            words = [self.stemmer.stem(word) for word in words]
        return words

    def load_documents(self):
        '''
        Load documents from the directory, preprocess them, and build the ground truth (filename -> author)
        
        Returns:
            documents: A dict mapping each document filename to its preprocessed text.
            ground_truth: A list of tuples, where each tuple contains a document filename and its author.
        '''
        documents = {}
        ground_truth = []
        for split in ['C50train', 'C50test']:
            split_path = os.path.join(self.base_dir, split)
            for author in os.listdir(split_path):
                author_path = os.path.join(split_path, author)
                if os.path.isdir(author_path):
                    for filename in os.listdir(author_path):
                        file_path = os.path.join(author_path, filename)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            text = file.read()
                            processed_text = self.preprocess_text(text)
                            documents[filename] = processed_text
                            ground_truth.append((filename, author))
        return documents, ground_truth

    def compute_okapi_lookup(self, k1=1.5, b=0.75, k2=100):
        '''
        Compute the Okapi lookup table for all documents
        Expected use: self.okapi_lookup[doc1_idx][doc2_idx] -> Okapi similarity between doc1 and doc2
        '''
        num_docs = len(self.documents)
        doc_lengths = {doc: len(terms) for doc, terms in self.documents.items()}
        avg_doc_length = sum(doc_lengths.values()) / num_docs
        self.okapi_lookup = np.zeros((num_docs, num_docs), dtype=np.float32)
        pbar = tqdm(range(num_docs))
        for d1_idx, (doc1, doc1_terms) in enumerate(self.doc_term_counts.items()):
            for d2_idx, (doc2, doc2_terms) in enumerate(self.doc_term_counts.items()):
                similarity = 0
                for term, term_count in doc1_terms.items():
                    if term in doc2_terms:
                        df = self.term_doc_freq[term] / num_docs
                        idf = np.log((num_docs - df + 0.5) / (df + 0.5))
                        doc1_score = idf * ((k1 + 1) * term_count) / (k1 * (1 - b + (b * doc_lengths[doc1]) / avg_doc_length) + term_count)
                        doc1_score *= ((k2 + 1) * doc2_terms[term]) / (k2 + doc2_terms[term])
                        similarity += doc1_score
                        if similarity == np.nan:
                            print("nan")
                self.okapi_lookup[d1_idx][d2_idx] = similarity
            pbar.update(1)                

    def compute_tfidf(self, min_df=5, max_df=0.8):
        '''
        Compute the TF-IDF matrix for the loaded documents
        
        Args:
            min_df (int): The minimum document frequency required for a term to be included.
            max_df (float): The maximum document frequency proportion allowed for a term to be included.
        '''
        term_doc_freq = defaultdict(int)
        doc_term_counts = defaultdict(Counter)

        # Efficiently count term occurrences and document frequencies
        for doc, words in self.documents.items():
            term_counts = doc_term_counts[doc]
            for term in words:
                term_counts[term] += 1
            for term in set(words):
                term_doc_freq[term] += 1

        num_docs = len(self.documents)

        # Filter vocabulary based on df thresholds
        vocab = {term for term, df in term_doc_freq.items() if df >= min_df and df / num_docs <= max_df}
        vocab = sorted(vocab)  # Sorted for consistency
        self.vocab = vocab
        vocab_index = {term: idx for idx, term in enumerate(vocab)}

        # Precompute the IDF for all terms in the vocabulary
        idf = np.zeros(len(vocab))
        for idx, term in enumerate(vocab):
            idf[idx] = np.log(num_docs / (1 + term_doc_freq[term]))  # Vectorized IDF computation

        # Create a term-document matrix for the TF computation
        term_matrix = np.zeros((num_docs, len(vocab)), dtype=np.float32)

        # Fill in the term-document matrix with term frequencies
        for doc_idx, (doc, term_counts) in enumerate(doc_term_counts.items()):
            total_terms = sum(term_counts.values())
            for term, count in term_counts.items():
                if term in vocab_index:
                    term_idx = vocab_index[term]
                    term_matrix[doc_idx, term_idx] = count / total_terms  # Compute TF
        self.doc_term_counts = doc_term_counts
        self.term_doc_freq = term_doc_freq

        # Compute the final TF-IDF matrix (TF * IDF)
        self.tfidf_matrix = term_matrix * idf  # Vectorized multiplication of TF and IDF

    # Save TF-IDF matrix, ground truth, okapi tables
    def save_output(self, output_file, ground_truth_file):
        '''
        Save the computed TF-IDF matrix (numpy array) and the ground truth csv
        '''
        np.save(output_file, self.tfidf_matrix)
        
        with open(ground_truth_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Author'])
            writer.writerows(self.ground_truth)


        np.save("okapi_lookup.npy", self.okapi_lookup)

        print(f"Saved TF-IDF matrix to {output_file} and ground truth to {ground_truth_file}")


if __name__ == "__main__":
    # try it with `python textVectorizer.py C50/ matrix.npy ground_truth.csv --stemming --stopwords`
    parser = argparse.ArgumentParser(description="Vectorize Reuters 50-50 dataset using TF-IDF from scratch")
    parser.add_argument('dataset_path', type=str, help="Path to the root Reuters 50-50 dataset directory")
    parser.add_argument('output_file', type=str, help="Path to save TF-IDF matrix (NumPy .npy file)")
    parser.add_argument('ground_truth_file', type=str, help="Path to save ground truth CSV")
    parser.add_argument('--stemming', action='store_true', help="Enable stemming")
    parser.add_argument('--stopwords', action='store_true', help="Enable stopword removal")


    args = parser.parse_args()
    
    tv = textVectorizer(args.dataset_path, remove_stopwords=args.stopwords, apply_stemming=args.stemming)
    print("Loading documents...")
    tv.load_documents()
    print(f"Loaded {len(tv.documents)} documents.")
    
    print("Computing TF-IDF...")
    tv.compute_tfidf()
    print(f"Generated TF-IDF matrix.")
    
    print("Computing Okapi tables...")
    tv.compute_okapi_lookup()
    print(f"Generated Okapi tables.")

    print("Saving outputs...")
    tv.save_output(args.output_file, args.ground_truth_file)
    print("Processing complete.")
