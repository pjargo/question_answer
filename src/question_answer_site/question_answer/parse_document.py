from langdetect import detect_langs
import fitz
import hashlib
import nltk
from nltk.corpus import wordnet
import os
import pandas as pd
import regex
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .mongodb import MongoDb
from .utils import remove_non_text_elements, clean_text, remove_non_word_chars, deal_with_line_breaks_and_hyphenations, \
    tokens_to_embeddings
from .config import username, password, cluster_url, database_name
from urllib.parse import quote_plus
from gibberish_detector import detector
from gibberish_detector import trainer
import urllib.request as req
from transformers import BertTokenizer

bert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

target_url = 'https://raw.githubusercontent.com/rrenaud/Gibberish-Detector/master/big.txt'
file = req.urlopen(target_url)
data = ' '.join([line.decode('utf-8') for line in file])
Detector = detector.Detector(
    trainer.train_on_content(data, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'), threshold=4.0)

# Escape the username and password
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)


def update_collection(collection: str, parsed_data):
    """
    Update Mongodb collection with data extracted from added file

    :param collection: (str) mongodb collection name. either "extracted_text" or "parsed_documents"
    :param parsed_data: ([dict]) data from new document
    :return: None
    """
    mongodb = MongoDb(escaped_username, escaped_password, cluster_url, database_name,
                      collection_name=collection)
    if mongodb.connect():
        print(f"Updating the '{collection}' collection")
        doc_cnt = mongodb.count_documents()
        print(f"{doc_cnt} documents in '{collection}' before adding")

        # Insert the JSON data as a document into the collection
        document_tracker = set()
        never_need = ['language', 'language_probability', 'Path', 'token_embeddings', 'chunk_text',
                      'chunk_text_less_sw']
        parsed_need = ['counter', 'token_embeddings_less_sw', 'tokens_less_sw', 'tokens']
        extracted_need = ['Original_Text', 'Text']
        for data_obj in parsed_data:
            # Update extracted_text collection
            if data_obj['Document'] not in document_tracker and collection == "extracted_text":
                for key in never_need + parsed_need:
                    data_obj.pop(key)

                document_tracker.add(data_obj['Document'])
                mongodb.insert_document(data_obj)  # Add the data to the mongo collection

            # Update parsed_documents collection
            elif collection == "parsed_documents":
                for key in never_need + extracted_need:
                    data_obj.pop(key)
                mongodb.insert_document(data_obj)

        doc_cnt = mongodb.count_documents()
        print(f"{doc_cnt} documents in '{collection}' after adding")
    mongodb.disconnect()  # Close mongo client


def get_sha256(content):
    """
    Gets the sha256 code from the content of the entry

    :param content: (str) text
    :return: (string) sha256 code
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_text(filepath):
    """
    Gets raw text from pdf file

    :param filepath: (str) the path to the pdf file
    :return: (str, str) returns the document text, the abstract of the text, and the language of the text
    """
    if filepath.endswith('.pdf'):
        doc = fitz.open(filepath)
        full_text = ''.join(page.get_text('text') for page in doc)
        doc.close()
        return full_text

    elif filepath.endswith('.txt') or filepath.endswith('csv'):
        with open(filepath, 'r') as file:
            full_text = ''.join(line.strip() for line in file)
            return full_text
    print('File not a recognized format')
    return ''


def get_language(text):
    """
    Identfy language of the text

    :param text: (str)
    :return: language detected
    """
    try:
        languages = detect_langs(text)
    except:
        print('\tlanguage detection error!')
        languages = ''

    return languages


def parse_document(directory, embedding_layer_model=None, tokenizer=bert_base_tokenizer,
                   chunk_size=100, chunk_overlap=0, additional_stopwords=None):
    """
    Parse PDF document or directory containing pdf document(s)

    :param directory: (str) File path or directory path
    :param embedding_layer_model:
    :param tokenizer:
    :param chunk_size:
    :param chunk_overlap:
    :param additional_stopwords:
    :return:
    """
    # Directory or file
    if additional_stopwords is None:
        additional_stopwords = []
    try:
        # Will cause exception if directory is a single file
        all_docx = os.listdir(directory)
    except NotADirectoryError:
        all_docx = [os.path.basename(directory)]
        directory = os.path.dirname(directory)

    # Get a list of dictionaries containing all the desired information, one for each pdf
    docs = []
    for filename in all_docx:
        # Get text and languages
        filepath = os.path.join(directory, filename)
        print(filepath, filename)
        text = get_text(filepath)
        langs = get_language(text)

        # put text in dictionary
        pdf_dict = text_to_dict(text, langs)
        pdf_dict['Document'], pdf_dict['Path'] = filename, filepath
        if pdf_dict == {}:
            continue

        docs.append(pdf_dict)

    # Put list of dictionaries into dataframe
    pdfs_df = pd.DataFrame(docs)
    print('processing text...')
    pdfs_df = cleanup(pdfs_df, col='Text', replace_math=False, replace_numbers=False,
                      check_pos=False, remove_meta_data=False, remove_punct=False,
                      add_acronym_periods=False, lemmatize=False, stem=False,
                      remove_non_alpha=False, remove_SW=False, remove_gibberish=True,
                      remove_non_text=True, remove_spec_chars=True, remove_line_breaks=True,
                      remove_unicode=True)

    # Tokenize the dataframe of extracted and cleaned text
    pdfs_df = tokenize_df_of_texts(pdfs_df, tokenizer=tokenizer, REMOVE_SW_COL=False, additional_stopwords=[])

    # Chunk the Tokenized dataframe
    parsed_list = chunk_df_of_tokens(pdfs_df, chunk_size=chunk_size, embedding_model=embedding_layer_model,
                                     overlap=chunk_overlap, additional_stopwords=additional_stopwords,
                                     tokenizer=tokenizer)

    # Don't add document if it already exists, Add counter
    max_cnter = 0  # Get the largest value in the storage directory if json files are already in there
    sha_set = set()
    mongo_docs = get_all_counter_sha()
    for doc in mongo_docs:
        max_cnter = max(max_cnter, doc['counter'])
        sha_set.add(doc['sha_256'])

    # dict_keys(['chunk_text', 'chunk_text_less_sw', 'tokens', 'tokens_less_sw', 'token_embeddings',
    # 'token_embeddings_less_sw', 'Document', 'Path', 'Text', 'Original_Text', 'sha_256', 'language',
    # 'language_probability', 'counter'])
    chunck_dict_list = []
    for chunk_dict in parsed_list:
        if chunk_dict['sha_256'] in sha_set:
            continue
        max_cnter += 1
        chunk_dict['counter'] = max_cnter
        chunck_dict_list.append(chunk_dict)

    return chunck_dict_list


def text_to_dict(text, langs):
    """
    Generate dictionary of information from PDF

    :param text: (str) Extracted text from document
    :param langs:
    :return: (dict) Information later to be added to dataframe
    """
    if text == '':
        return {}

    language_prob = 0.0 if langs == '' else langs[0].prob
    language = '' if langs == '' else langs[0].lang

    sha256 = get_sha256(text)  # Get sha256

    section_dict = {'Document': '',
                    'Path': '',
                    'Text': text,
                    'Original_Text': text,
                    'sha_256': sha256,
                    'language': language,
                    'language_probability': language_prob,
                    'chunk_text': '',
                    'chunk_text_less_sw': '',
                    'tokens': [],
                    'tokens_less_sw': [],
                    'token_embeddings': [],
                    'token_embeddings_less_sw': []}

    return section_dict


def tokenize_df_of_texts(df, tokenizer=bert_base_tokenizer, REMOVE_SW_COL=False, additional_stopwords=[]):
    """
    Use BERT tokenizer to tokenize a dataframe of extracted text

    :param additional_stopwords:
    :param REMOVE_SW_COL:
    :param df: (pandas.DataFrame) dataframe with 'TEXT' column
    :param tokenizer: (BertTokenizer.from_pretrained('bert-base-uncased'))
    :return: (pandas.DataFrame) modified dataframe
    """

    print("tokenize the processed text...")

    texts = df["Text"].tolist()
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    df['tokens'] = tokenized_texts

    if REMOVE_SW_COL:
        nltk_stop_words = nltk.corpus.stopwords.words('english')
        nltk_stop_words = nltk_stop_words + ["Ġ" + word for word in nltk_stop_words]
        if additional_stopwords:
            nltk_stop_words.extend(additional_stopwords)
        nltk_stop_words = set(nltk_stop_words)

        tokenized_texts_less_sw = [[token for token in tokens_list if token not in nltk_stop_words] for tokens_list in
                                   tokenized_texts]
        df['tokens_less_sw'] = tokenized_texts_less_sw

    return df


def chunk_df_of_tokens(df, chunk_size=100, embedding_model=None, overlap=0, additional_stopwords=[],
                       tokenizer=bert_base_tokenizer):
    """
    Chunk the dataframe that has previously been tokenized

    :param tokenizer:
    :param df: (pandas.DataFrame)
    :param chunk_size: (int) Size of chunk we split or pad list of tokens into
    :param embedding_model: put tokens into their word embeddings
    :param overlap: (int) By how many tokens we want the chunks to overlap
    :return: Pandas.DataFrame tranformed then put into a dictionary - a list of dictionary with the following keys: Document, Abstract, Text, Abstract_Original, Original_Text, Path, sha_256, laguage, language_probability, Authors, Tilte, url, date
    """
    print("Chunking the tokenized text...")
    chunked_df = get_chunked_df(df, chunk_size=chunk_size, embedding_model=embedding_model, overlap=overlap,
                                additional_stopwords=additional_stopwords, tokenizer=tokenizer)
    print("\nprinting the shape of chunked dataframe")
    print(chunked_df.shape)

    return chunked_df.T.to_dict().values()


def get_chunked_df(df, chunk_size=100, embedding_model=None, overlap=0, additional_stopwords=[],
                   tokenizer=bert_base_tokenizer):
    """
    Chunk the 'tokens' column in the dataframe and optionally get the embeddings of the tokens

    :param tokenizer:
    :param df: (pandas.DataFrame) dataframe of parsed pdf documents
    :param chunk_size: Number of tokens per chunk
    :param model: (optional) Embedding Layer
    """
    # Create a new DataFrame to store the chunked data
    chunked_data = []
    # Iterate over each row in the original DataFrame
    for _, row in df.iterrows():
        # Get the tokens and metadata for the current row
        tokens = row["tokens"]
        metadata = row.drop("tokens").drop("token_embeddings").drop("token_embeddings_less_sw") \
            .drop("tokens_less_sw").drop("chunk_text").drop(
            "chunk_text_less_sw")  # Drop the "tokens" column from the metadata

        # Chunk the tokens into sequences of length 100
        chunked_tokens, chunked_tokens_less_sw, chunked_text, chunked_text_less_sw = chunk_tokens(tokens,
                                                                                                  max_chunk_length=chunk_size,
                                                                                                  overlap=overlap,
                                                                                                  additional_stopwords=additional_stopwords,
                                                                                                  tokenizer=tokenizer)

        # Pad the sequences less than 100 tokens, don't need to pad the chunked token less stop words. Only for
        # candidate search
        padded_tokens = [token_list + ["[PAD]"] * (100 - len(token_list)) for token_list in chunked_tokens]

        if embedding_model:
            embedded_tokens = [tokens_to_embeddings(token_list, embedding_model, RANDOM=False) for token_list
                               in padded_tokens]
            embedded_tokens_less_sw = [tokens_to_embeddings(token_list, embedding_model, RANDOM=False) for
                                       token_list in chunked_tokens_less_sw]
        else:
            embedded_tokens = [[] for token_list in padded_tokens]
            embedded_tokens_less_sw = [[] for token_list in chunked_tokens_less_sw]

        # Create new rows for each chunked and padded tokens along with metadata
        for padded_tokens_chunk, embedded_tokens_chunk, tokens_chunk_less_sw, embedded_tokens_chunk_less_sw, chunk_text, chunk_text_less_sw in \
                zip(padded_tokens, embedded_tokens, chunked_tokens_less_sw, embedded_tokens_less_sw, chunked_text,
                    chunked_text_less_sw):
            new_row = {"chunk_text": chunk_text,
                       "chunk_text_less_sw": chunk_text_less_sw,
                       "tokens": padded_tokens_chunk,
                       "tokens_less_sw": tokens_chunk_less_sw,
                       "token_embeddings": embedded_tokens_chunk,
                       "token_embeddings_less_sw": embedded_tokens_chunk_less_sw,
                       **metadata}
            chunked_data.append(new_row)

    # Create the new DataFrame with the chunked and padded data
    chunked_df = pd.DataFrame(chunked_data)

    # Return he resulting DataFrame with chunked and padded tokens and metadata
    return chunked_df


def chunk_tokens(tokens, max_chunk_length=100, overlap=0, additional_stopwords=[], tokenizer=bert_base_tokenizer):
    """

    :param tokens: ([int]) A list of integers representing the token ID's
    :param max_chunk_length: (int) The length of the sequence of each chunk
    :param overlap: By how many tokens
    :param additional_stopwords: ['from', 'subject', 're', 'edu', 'use', 'table', 'figure', 'arxiv', 'sin', 'cos', 'tan', 'log', 'fx', 'ft', 'dx', 'dt', 'xt']
    """
    nltk_stop_words = nltk.corpus.stopwords.words('english')
    nltk_stop_words += ["Ġ" + word for word in nltk_stop_words]
    if additional_stopwords:
        nltk_stop_words.extend(additional_stopwords)
    nltk_stop_words = set(nltk_stop_words)

    chunked_tokens = []
    chunked_tokens_less_sw = []

    chunk_text = []
    chunk_text_less_sw = []

    current_chunk = []
    current_length = 0
    i = 0

    while i < len(tokens):
        token = tokens[i]
        current_chunk.append(token)
        current_length += 1
        if current_length >= max_chunk_length:
            chunked_tokens.append(current_chunk)

            current_chunk_less_sw = [t for t in current_chunk if t not in nltk_stop_words]
            chunked_tokens_less_sw.append(current_chunk_less_sw)

            current_chunk_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_chunk))
            chunk_text.append(current_chunk_text)
            current_chunk_text_less_sw = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_chunk_less_sw))
            chunk_text_less_sw.append(current_chunk_text_less_sw)

            current_chunk = []
            current_length = 0
            i -= overlap  # Intentional overlap of the token arrays
        i += 1

    if current_chunk:
        chunked_tokens.append(current_chunk)
        current_chunk_less_sw = [t for t in current_chunk if t not in nltk_stop_words]
        chunked_tokens_less_sw.append(current_chunk_less_sw)

        current_chunk_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_chunk))
        chunk_text.append(current_chunk_text)
        chunk_text_less_sw.append(current_chunk_text)

    return chunked_tokens, chunked_tokens_less_sw, chunk_text, chunk_text_less_sw


def get_all_counter_sha():
    """
    Get all document counter and sha256 values from mongo collection

    :return: ([dict])
    """
    collection_name = "parsed_documents"
    # collection_name = "extracted_text"
    cursor = list()
    mongodb = MongoDb(escaped_username, escaped_password, cluster_url, database_name, collection_name)
    if mongodb.connect():
        cursor = mongodb.get_collection().find({}, {"counter": 1, "sha_256": 1, "_id": 0})

    return list(cursor)


def cleanup(df, col='Text', replace_math=False,
            replace_numbers=False, check_pos=False,
            remove_meta_data=True, remove_punct=False,
            add_acronym_periods=False, lemmatize=False,
            stem=False, remove_non_alpha=False,
            remove_SW=False, remove_gibberish=True,
            remove_non_text=True, remove_spec_chars=True,
            remove_line_breaks=True, remove_unicode=True):
    """
    Clean text according to input parameters
    Input: df with intended text in "Text" column, as space-delimited string of words (Standard sentence)
    """
    df[col] = df[col].apply(lambda x: x.replace('-\n', ''))
    df[col] = df[col].apply(lambda x: x.replace('\n', ' '))

    if replace_math:
        print('removing math symbols and letters...')
        df[col] = df[col].apply(
            lambda z: ' '.join(['' if regex.search(r'\p{Greek}|\p{S}', word) else word for word in z.split()]))

    if replace_numbers:
        print('removing stand allow digits (or figrue table numbers)...')
        df[col] = df[col].apply(lambda z: ' '.join(['' if re.search(r'\d', word) else word for word in z.split()]))

    if check_pos:
        print('keep only nouns...')
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        # Nouns only
        tag_dict = {'N': wordnet.NOUN}

        df[col] = df[col].swifter.apply(lambda z: ' '.join(
            [word if tag_dict.get(nltk.pos_tag([word])[0][1][0].upper(), False) else '' for word in z.split()]))

    if remove_meta_data:
        print('removing meta data...(Names, Dates, Places...)')

    if remove_punct:
        print('removing punctuation...')
        df[col] = df[col].apply(
            lambda x: x.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace('  ',
                                                                                                            ' ').strip())

    if add_acronym_periods:
        # Add periods to any capital letter sequence since thats probably an acronym (This is for normalization package)
        print('adding acronym periods...')
        df[col] = df[col].apply(lambda s: ' '.join([''.join([c + "." if c.isalpha() else c for c in w])
                                                    if w.isupper()
                                                    else w
                                                    for w in s.split()]))
    if remove_non_alpha:
        print('removing words with chars not a-zA-Z0-9...')
        # df[col] = df[col].apply(lambda x: ' '.join(['' if re.search(r'[^a-zA-Z\d\s]', word) else word for word in x.split()]))
        df[col] = df[col].apply(remove_non_word_chars)

    # Lowercase
    print('making lower-case...')
    df[col] = df[col].map(lambda x: x.lower())

    # It likes the stopwords to be lowercase before removing them
    if remove_SW:
        print('removing stop words...')
        nltk_stop_words = nltk.corpus.stopwords.words('english')
        nltk_stop_words.extend(
            ['from', 'subject', 're', 'edu', 'use', 'table', 'figure', 'arxiv', 'sin', 'cos', 'tan', 'log', 'fx', 'ft',
             'dx', 'dt', 'xt'])
        # Tokenized text with standard stopwords and punct removed
        # TODO: CHANGE THE TOKENIZATION
        df[col] = df[col].swifter.apply(lambda z: ' '.join([t for t in word_tokenize(z)
                                                            if t not in nltk_stop_words]))
        print('stop words removed.\n')

    if remove_non_text:
        print('Removing non-text elements (extra whitespaces)...')
        df[col] = df[col].apply(clean_text)

    if remove_spec_chars:
        print('Removing unnecessary whitespace and special characters...')
        df[col] = df[col].apply(remove_non_text_elements)

    if remove_line_breaks:
        print('Removing line breaks...')
        df[col] = df[col].apply(deal_with_line_breaks_and_hyphenations)

    if lemmatize:
        print('lemmitizing...')
        lemmatizer = WordNetLemmatizer()
        df[col] = df[col].swifter.apply(
            lambda z: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in z.split()]))

    if stem:
        print("stemming...")
        # df.Text, stem_dict = stem_words(df.Text)

    if remove_gibberish:
        print('Removing gibberish...')
        df[col] = df[col].apply(lambda z: ' '.join(['' if Detector.is_gibberish(word) else word for word in z.split()]))

    if remove_unicode:
        print('Removing unicode...')
        # df[col] = df[col].apply(lambda text: normalize_text(text).encode('unicode-escape').decode())
        # df[col] = df[col].apply(lambda x: ' '.join([normalize_text(word) for word in x.split()]))

    print('remove single letters or super large words (so big they don\'t make sense)...')
    df[col] = df[col].apply(lambda z: ' '.join([word for word in z.split() if len(word) < 20 and len(word) > 1]))
    print('done cleaning.\n')
    return df


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts

    Parameter
    ---------
    word (str) - word to find POS

    Returns
    -------
    str - the POS tag
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
