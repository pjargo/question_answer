from langdetect import detect_langs
from nltk.stem import PorterStemmer, SnowballStemmer 
import gensim
import fitz
import re
import hashlib
import spacy
nlp = spacy.load('en_core_web_lg')
import nltk
from nltk.corpus import wordnet
import os
import pandas as pd
import regex
import re
import spacy
import swifter
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import unicodedata
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


from gibberish_detector import detector
from gibberish_detector import trainer
import urllib.request as req
target_url = 'https://raw.githubusercontent.com/rrenaud/Gibberish-Detector/master/big.txt'
file = req.urlopen(target_url)
data = ' '.join([line.decode('utf-8') for line in file])
Detector = detector.Detector(trainer.train_on_content(data, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'), threshold=4.0)

from tqdm import tqdm  # For progress bar
import numpy as np

def get_sha256(content):
    '''
    Gets the sha256 code from the content of the entry
    
    Args:
        content : the end of week entry content, specifically a string
        
    Returns:
        string : sha256 code
    '''
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_document_date(filepath, is_arxiv=False):
    if filepath.endswith('.pdf'):

        if is_arxiv:
            doc = fitz.open(filepath)
            page1 = doc[0].get_text('text')
            dates = []
            for p in page1.split('\n'):
                if p.startswith('arXiv'):
                    for ent in entity_finder(p):
                        if ent['label'] == "DATE":
                            try:
                                date = parser.parse(ent['text'])
                            except:
                                continue

                            if date < datetime.now():
                                dates.append(date)
                    break
            
        else:
            text = get_title_text(filepath)
            dates = []
            for ent in entity_finder(text):
                if ent['label'] == "DATE":
                    try:
                        date = parser.parse(ent['text'])
                    except:
                        continue
                    
                    if date < datetime.now():
                        dates.append(date)

        if dates == []:
            if is_arxiv:
                return get_document_date(filepath, is_arxiv=False)
            else:
                return ''
        else:
            return min(dates, key=lambda x: abs(x - datetime.now())).strftime('%m/%d/%Y')


def get_title_text(filepath):
    if not filepath.endswith('.pdf'):
        print('File path not a pdf')
        return None
    
    doc = fitz.open(filepath)

    full_text = ''
    page1 = ''
    for i, page in enumerate(doc):
        full_text += page.get_text('text')
        if i == 0:
            page1 = full_text
        
    re_section = r'ABSTRACT|Abstract|Summary|SUMMARY' 
    p = re.compile(re_section)
    indices = [m.start() for m in p.finditer(full_text)]
    if len(indices) > 0:
        i_start = indices[0]
        text = full_text[:i_start]
    else:
        print('\thas no \'Abstract\' section.')
        re_section = r'INTRODUCTION|Introduction|\n1\s' 
        p = re.compile(re_section)
        indices = [m.start() for m in p.finditer(full_text)]
        if len(indices) > 0:
            i_start = indices[0]
            text = full_text[:i_start]
        else:
            text = page1
    
    return text


def get_text(filepath, ru_model_path=None, zh_model_path=None):
    '''
    Gets raw text from pdf file. Will translate russian and chinese documents to english.

    Parameters
    ----------
    filepath (str) - the path to the pdf file
    ru_model_path (str, None) - the path to the model to translate russian to english (default = None).  If None, model loaded from "language_parsers" directory.
    zh_model_path (str, None) - the path to the model to translate chinese to english (default = None). If None, model loaded from "language_parsers" directory.

    Returns
    --------
    str, str, str - returns the document text, the abstract of the text, and the language of the text
    '''
    if not filepath.endswith('.pdf'):
        print('File path not a pdf')
        return None
    
    doc = fitz.open(filepath)

    full_text = ''
    for page in doc:
        full_text += page.get_text('text')
    
    try:
        langs = detect_langs(full_text)
        language = langs[0].lang
        if language != 'en':
            print('not english:', language)
            if language == 'zh-cn':
                print('\ttranslating zh to en...')
                full_text = zh_to_en(zh_model_path, full_text)
                print('\ttranslated.')
            elif language == 'ru':
                print('\ttranslating ru to en...')
                full_text = ru_to_en(ru_model_path, full_text)
                print('\ttranslated.')
            else:
                print('translation model to english not created...yet')
                return '', '', langs
    except:
        print('\tlanguage detection error!')
        langs = ''

    re_section = r'ABSTRACT|Abstract|Summary|SUMMARY' 
    p = re.compile(re_section)
    indices = [m.end() for m in p.finditer(full_text)]
    if len(indices) > 0:
        i_start = indices[0]
    else:
        i_start = 0
        print('\thas no \'Abstract\' section.')
        
    re_section = r'REFERENCE|Reference|Bibliography|BIBLIOGRAPHY|ACKNOWLEDGMENTS|Acknowledgments' 
    p = re.compile(re_section)
    indices = [m.start(0) for m in p.finditer(full_text)]
    if len(indices) > 0:
        i_end = indices[-1]
    else:
        i_end = -1
        print('\thas no \'References\' section.')
        
    re_section = r'INTRODUCTION|Introduction|\n1\s' 
    p = re.compile(re_section)
    indices = [m.start(0) for m in p.finditer(full_text)]

    if len(indices) > 0:
        i_end_abstract = indices[0]
    else:
        i_end_abstract = i_start
        print('\thas no \'introduction\' section.')

    return full_text[i_start:i_end], full_text[i_start:i_end_abstract], langs


def stem_words(list_text):
    '''
    Stem words in corpus.

    Parameters
    -----------
    list_text (list of str) - list of texts representing the corpus

    Returns
    -------
    list of str - the list of the text with words stemmed from "list_text"
    '''
    print('stemming words...')
    stem_dict = {}
    snowball = SnowballStemmer("english")
    for i in range(len(list_text)):
        temp = []
        for w in list_text[i].split():
            stemmed = snowball.stem(w)
            temp.append(stemmed)
            if stemmed in stem_dict.keys():
                if w not in stem_dict[stemmed]:
                    stem_dict[stemmed].append(w)
            else:
                stem_dict[stemmed] = [w]
        list_text[i] = ' '.join(temp).replace('  ', ' ')
    print('words stemmed.\n')
    return list_text


def ngram(list_text, n=2, min_count=5, threshold=100):
    '''
    N-grams words from a corpus

    Parameters
    ----------
    list_text (list of str) - the list of texts that represent the corpus
    n (int) - the number of "grams" allowed, for example, 2 = bigram and 3 = trigram ... (default = 2)
    min_count (int) - the minimum number of times a pharse has to show up to be considered a phrase (defualt = 5).
    threshold (int) - Represent a score threshold for forming the phrases (higher means fewer phrases). A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold. Heavily depends on concrete scoring-function, see the scoring parameter. (default = 100)
    
    Returns
    -------
    list of str - the list of texts with the "ngrammed" words
    '''
    print('ngramming words...')
    numbers = {2:'bigram', 3:'trigram', 4:'quadgram'}
    texts = [sentence.split(' ') for sentence in list_text]
    init_gram = {}
    for i in range(2,n+1):
        try:
            print(numbers[i], 'model started...')
        except:
            print('%s-gram model started...')
        ngram = gensim.models.Phrases(texts, min_count=min_count, threshold=threshold)
        ngram_mod = gensim.models.phrases.Phraser(ngram)
        texts = [ngram_mod[doc] for doc in texts]
           
        try:
            print(numbers[i], 'model finished.\n')
        except:
            print('%s-gram model finished.\n')
            
        init_gram[i] = {'ngram':ngram, 'ngram_mod':ngram_mod, 'texts':texts}
    
    return [' '.join(text) for text in texts]


def entity_finder(text):
    # 2.6 Entity Names Annotation
    # Names (often referred to as “Named Entities”) are annotated according to the following
    # set of types:
    # PERSON People, including fictional
    # NORP Nationalities or religious or political groups
    # FACILITY Buildings, airports, highways, bridges, etc.
    # ORGANIZATION Companies, agencies, institutions, etc.
    # GPE Countries, cities, states
    # LOCATION Non-GPE locations, mountain ranges, bodies of water
    # PRODUCT Vehicles, weapons, foods, etc. (Not services)
    # EVENT Named hurricanes, battles, wars, sports events, etc.
    # WORK OF ART Titles of books, songs, etc.
    # LAW Named documents made into laws 
    #  OntoNotes Release 5.0
    # 22
    # LANGUAGE Any named language
    # The following values are also annotated in a style similar to names:
    # DATE Absolute or relative dates or periods
    # TIME Times smaller than a day
    # PERCENT Percentage (including “%”)
    # MONEY Monetary values, including unit
    # QUANTITY Measurements, as of weight or distance
    # ORDINAL “first”, “second”
    # CARDINAL Numerals that do not fall under another typ
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in ['CARDINAL', 'PERCENT', 'MONEY', 'ORDINAL', 'QUANTITY'] or '\n' in ent.text:
            continue
        else:
            ent_chars = {'text': ent.text, # The str of the named entity phrase.
                         'start': ent.start_char, # Source str index of the first char.
                         'end': ent.end_char, # Source str index of the last+1 char.
                         'label': ent.label_} # A str label for the entity type.
            #print(ent_chars)
            ents.append(ent_chars)
    return ents


def removing_meta_data(nlp, text, remove_tags=['PERSON', 'FAC', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LAW', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']):
    '''
    Removes meta data from text.

    Parameters
    ----------
    nlp (spacy.en_core) - the spacy nlp model to find english tags/meta data
    text (str) - text of which needs meta data removed
    remove_tags (list) - list of entity name annotations/tags that are to be removed from "text" (default = ['PERSON', 'FAC', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LAW', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])

    Return
    ------
    str - the text with removed meta data

    Notes
    -----
    # 2.6 Entity Names Annotation
    # Names (often referred to as “Named Entities”) are annotated according to the following
    # set of types:
    # PERSON People, including fictional
    # NORP Nationalities or religious or political groups
    # FACILITY Buildings, airports, highways, bridges, etc.
    # ORGANIZATION Companies, agencies, institutions, etc.
    # GPE Countries, cities, states
    # LOCATION Non-GPE locations, mountain ranges, bodies of water
    # PRODUCT Vehicles, weapons, foods, etc. (Not services)
    # EVENT Named hurricanes, battles, wars, sports events, etc.
    # WORK OF ART Titles of books, songs, etc.
    # LAW Named documents made into laws 
    #  OntoNotes Release 5.0
    # 22
    # LANGUAGE Any named language
    # The following values are also annotated in a style similar to names:
    # DATE Absolute or relative dates or periods
    # TIME Times smaller than a day
    # PERCENT Percentage (including “%”)
    # MONEY Monetary values, including unit
    # QUANTITY Measurements, as of weight or distance
    # ORDINAL “first”, “second”
    # CARDINAL Numerals that do not fall under another typ
    '''
        
    doc = nlp(text)
    #str_tags = [(X.text, X.label_) for X in doc.ents]
    removed = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and len(ent.text.strip().split(' ')) < 2:
            continue
        elif ent.label_ in remove_tags:
            ent_chars = {'text': ent.text, # The str of the named entity phrase.
                         'start': ent.start_char, # Source str index of the first char.
                         'end': ent.end_char, # Source str index of the last+1 char.
                         'label': ent.label_} # A str label for the entity type.
            text = text.replace(ent.text, ' ')
            removed.append(ent_chars)
    
    return text, removed


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


def parsed_pdf_to_json(directory, storage_dir='./parsed_cleaned_pdfs', stem=False, abstract=True, embedding_layer_model=None, chunk_overlap=0):
    '''
    Parses and cleans and stores jsons of a directory of pdfs by extracting from each: document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the pdf documents
    storage_dir (str, optional) - the directory that the jsons of parsed documents will be stored
    stem (bool, optional) - True means to stem the words and False is to leave as is (default = False)
    abstract (bool, optional) - True means to normalize the abstract (default = False)
    chunk_overlap (int, optional) -  chunking the documents overlap by this many tokens(default = 0)

    '''
    
    try:
        os.mkdir(storage_dir)
    except (FileExistsError):
        pass
    
    parsed_list = parse_clean_pdfs(directory,
        embedding_layer_model=embedding_layer_model,
        chunk_overlap=chunk_overlap)
        
    parsed_dict = {i : doc for i, doc in enumerate(parsed_list)}
    
    for fname, doc in parsed_dict.items():
        with open(os.path.join(storage_dir, str(fname)+'.json'), 'w') as j_file:
                json.dump(doc, j_file, indent=4)
                
def parse_clean_pdfs(directory, embedding_layer_model, chunk_overlap=0):
    '''
    Parses and cleans a directory of pdfs by extracting from each: document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the pdf documents
    embedding_layer_model (model) - 
    chunk_overlap (int, optional) -  chunking the documents overlap by this many tokens(default = 0)

    Returns
    -------
    list of dicts - a list of dictionary with the following keys: Document, Abstract, Text, Abstract_Original, Original_Text, Path, sha_256, laguage, language_probability, Authors, Tilte, url, date
    '''
    
    all_docx = os.listdir(directory)    
    docs = []
    
    # Get a list of dictionaries containing all the desired information, one for each pdf
    for filepath in all_docx:
        if filepath.endswith('.pdf'):
            print(os.path.join(directory, filepath))
            pdf_dict = pdf_to_dict(directory, filepath)
            if pdf_dict == {}:
                continue
            docs.append(pdf_dict)
    
    # Put each dictionary into a dataframe and clean it
    df = pd.DataFrame(docs)
    print('processing text...')
    df = cleanup(df, col='Text', replace_math=False, replace_numbers=False, 
                 check_pos=False, remove_meta_data=False, remove_punct=False, 
                 add_acronym_periods=False, lemmatize=False, stem=False, 
                 remove_non_alpha=False, remove_SW=False, remove_gibberish=True,
                 remove_non_text=True, remove_spec_chars=True, remove_line_breaks=True,
                 remove_unicode=True)
    
    print("tokenize the processed text")
    texts = df["Text"].tolist()
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    df['tokens'] = tokenized_texts
    
    print("Chunking the tokenized text")
    chunked_df = get_chunked_df(df, chunk_size=100, embedding_model=embedding_layer_model, overlap=chunk_overlap)
    # chunked_df = chunked_df.drop(columns=["Original_Text"])
    print("\nprinting the shape of chunked dataframe")
    print(chunked_df.shape)
            
    return chunked_df.T.to_dict().values()


def pdf_to_dict(directory, filename):
    '''
    Extracts document name, abstract, original text, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the document
    filename (str) - the name of the pdf file

    Returns
    -------
    dict - a dictionary with the following keys: Document, Abstract, Text, Abstract_Original, Original_Text, Path, sha_256, laguage, language_probability, Authors, Tilte, url, date
    '''
    filepath = os.path.join(directory, filename)
    text, abstract_text, langs = get_text(filepath)
    if text == '':
        return {}
    if langs == '':
        language = ''
        language_prob = 0.0
    else:
        language = langs[0].lang
        language_prob = langs[0].prob
    sha256 = get_sha256(text)
    date = get_document_date(filepath, is_arxiv=True)
    section_dict = {'Document': filename,
                    'Abstract': abstract_text,
                    'Text' : text,
                    'Abstract_Original': abstract_text,
                    'Original_Text' : text,
                    'Path' : filepath,
                    'sha_256': sha256,
                    'language' : language,
                    'language_probability' : language_prob,
                    'Authors' : '',
                    'Title' : '',
                    'url' : '',
                    'date' : date,
                    'tokens':[],
                    'token_embeddings':[]}
    
    return section_dict


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
        df[col] = df[col].apply(lambda z: ' '.join(['' if regex.search(r'\p{Greek}|\p{S}', word) else word for word in z.split()]))
        
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

        df[col] = df[col].swifter.apply(lambda z: ' '.join([word if tag_dict.get(nltk.pos_tag([word])[0][1][0].upper(), False) else '' for word in z.split()]))
    
    if remove_meta_data:
        print('removing meta data...(Names, Dates, Places...)')
        nlp = spacy.load("en_core_web_md")
        df[col] = df[col].swifter.apply(lambda x: removing_meta_data(nlp, x)[0])
        
    if remove_punct:
        print('removing punctuation...')
        df[col] = df[col].apply(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).replace('  ', ' ').strip())
        
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
        nltk_stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'table', 'figure', 'arxiv', 'sin', 'cos', 'tan', 'log', 'fx', 'ft', 'dx', 'dt', 'xt'])
        # Tokenized text with standard stopwords and punct removed
        #TODO: CHANGE THE TOKENIZATION
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
        lemmatizer=WordNetLemmatizer()
        df[col] = df[col].swifter.apply(lambda z: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in z.split()]))
        
    if stem:
        df.Text, stem_dict = stem_words(df.Text)

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


def remove_non_text_elements(text):
    # Regular expression to remove non-text elements (headers, footers, page numbers, etc.)
    text = re.sub(r'\b(Header|Footer|Page \d+)\b', '', text)
    return text


def clean_text(text):
    # Remove unnecessary whitespace, ...
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading and trailing whitespace


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


def chunk_tokens(tokens, max_chunk_length=100, overlap=0):
    """
    
    :param tokens: ([int]) A list of integers representing the token ID's
    "param max_chunk: (int) The lengeth of the sequence of each ch 
    """
    chunked_tokens = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        current_chunk.append(token)
        current_length += 1
        if current_length >= max_chunk_length:
            chunked_tokens.append(current_chunk)
            current_chunk = []
            current_length = 0
            i -= overlap   # Intential overlap of the token arrays
        i+=1
            
    if current_chunk:
        chunked_tokens.append(current_chunk)
    return chunked_tokens


def get_chunked_df(df, chunk_size=100, embedding_model=None, overlap=0):
    """
    Chunk the 'tokens' column in the dataframe and optionally get the embeddings of the tokens

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
        metadata = row.drop("tokens").drop("token_embeddings") # Drop the "tokens" column from the metadata

        # Chunk the tokens into sequences of length 100
        chunked_tokens = chunk_tokens(tokens, max_chunk_length=chunk_size, overlap=overlap)

        # Pad the sequences less than 100 tokens
        padded_tokens = [tokens + ["[PAD]"] * (100 - len(tokens)) for tokens in chunked_tokens]

        if embedding_model:
            embedded_tokens = [tokens_to_embeddings(tokens, embedding_model, RANDOM=False).tolist() for tokens in padded_tokens]
        else: 
            embedded_tokens = [[] for tokens in padded_tokens]

        # Create new rows for each chunked and padded tokens along with metadata
        for padded_tokens_chunk, embedded_tokens_chunk in zip(padded_tokens, embedded_tokens):
            new_row = {"tokens": padded_tokens_chunk, "token_embeddings": embedded_tokens_chunk, **metadata}
            chunked_data.append(new_row)

    # Create the new DataFrame with the chunked and padded data
    chunked_df = pd.DataFrame(chunked_data)

    # Return he resulting DataFrame with chunked and padded tokens and metadata
    return chunked_df



def tokens_to_embeddings(tokens_list, model, RANDOM=True):
    """
    Convert a list of tokens to an embeddings matrix
    
    :param tokens_list: list of tokens to be embedded
    :param model: Enbeddings Layer Model
    """
    
    # Initialize an array to store embeddings
    query_embeddings = []

    # Loop through tokens in the tokenized query
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

    # Convert the list of embeddings to a NumPy array
    query_embeddings = np.array(query_embeddings)

    return query_embeddings




