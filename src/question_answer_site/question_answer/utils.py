import gensim
import spacy
import re
import os
import unicodedata
import numpy as np
from gibberish_detector import detector
from gibberish_detector import trainer
import urllib.request as req
from spellchecker import SpellChecker
from transformers import BertTokenizer

# Set proxy information
proxy_url = "http://33566:wed@proxy-west.aero.org:8080"

# Set proxy environment variables
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

bert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

target_url = 'https://raw.githubusercontent.com/rrenaud/Gibberish-Detector/master/big.txt'
file = req.urlopen(target_url)
data = ' '.join([line.decode('utf-8') for line in file])
Detector = detector.Detector(
    trainer.train_on_content(data, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'), threshold=4.0)


def post_process_output(decoded_text):
    # Define a list of punctuation marks to consider
    punctuation_marks = ['.', ',', ';', ':', '!', '?']

    # Use regular expressions to find punctuation tokens with spaces before them
    pattern = r'(\w)\s?(' + r'|'.join(re.escape(p) for p in punctuation_marks) + r')\s'
    processed_text = re.sub(pattern, r'\1\2 ', decoded_text)

    return processed_text


def correct_spelling(word):
    spell = SpellChecker()
    # Your spelling correction logic
    corrected_word = spell.correction(word)
    return corrected_word if corrected_word else word


def tokens_to_embeddings(tokens_list, model, RANDOM=True):
    """
    Convert a list of tokens to an embeddings matrix
    
    :param tokens_list: list of tokens to be embedded
    :param model: Word2Vec or spaCy custom model
    :param RANDOM: Whether to use random embeddings for unseen tokens (default=True)
    :return: numpy array of embeddings for each input token
    """

    # Initialize an array to store embeddings
    query_embeddings = []

    # Check the type of the model
    if isinstance(model, gensim.models.Word2Vec):
        # Handle Word2Vec model
        for token in tokens_list:
            if token in model.wv:
                query_embeddings.append(model.wv[token])
            else:
                # Handle unseen tokens with random embeddings
                if RANDOM:
                    random_embedding = np.random.rand(model.vector_size)
                    query_embeddings.append(random_embedding)
                else:
                    zero_embedding = np.zeros(model.vector_size)
                    query_embeddings.append(zero_embedding)
    elif isinstance(model, spacy.language.Language):
        # Handle spaCy custom model
        for token in tokens_list:
            token = model(token)
            if token.has_vector:
                query_embeddings.append(token.vector.tolist())
            else:
                # Handle unseen tokens with random embeddings
                if RANDOM:
                    random_embedding = np.random.rand(model.vocab.vectors_length).tolist()
                    query_embeddings.append(random_embedding)
                else:
                    zero_embedding = np.zeros(model.vocab.vectors_length).tolist()
                    query_embeddings.append(zero_embedding)
    else:
        raise ValueError("Unsupported model type. Please provide a Gensim Word2Vec model or spaCy custom model.")

    # Convert the list of embeddings to a NumPy array
    # query_embeddings = np.array(query_embeddings)

    return query_embeddings


def remove_non_text_elements(text):
    # Regular expression to remove non-text elements (headers, footers, page numbers, etc.)
    text = re.sub(r'\b(Header|Footer|Page \d+)\b', '', text)
    return text


def clean_text(text):
    # Remove unnecessary whitespace, ...
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.lower().strip()  # Remove leading and trailing whitespace


def remove_non_word_chars(text):
    # Remove non-word characters (punctuation, etc.)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def deal_with_line_breaks_and_hyphenations(text):
    # Deal with line breaks (join words that are separated by line breaks)
    text = re.sub(r'(\S)-\n(\S)', r'\1\2', text)  # Join hyphenated words split across lines
    text = re.sub(r'\n', ' ', text)  # Replace line breaks with spaces

    return text


def normalize_text(text):
    # Normalize the text to NFC (Normal Form C)
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return normalized_text
