"""
Terry Cox
terry.cox@aero.org

"""

import os
from entity_extraction import get_document_date
import urllib.request as req
from datetime import date
import calendar
from gibberish_detector import trainer
import requests
from gibberish_detector import detector
req.getproxies()

#user your credentials and url to authenticate
proxy_user = 32856

my_date = date.today()
proxy_pass = calendar.day_name[my_date.weekday()].lower()[:3]

os.environ['http_proxy'] = "https://%s:%s@www-proxy-west.aero.org:8080"%(proxy_user, proxy_pass)
os.environ['https_proxy'] = "https://%s:%s@www-proxy-west.aero.org:8080"%(proxy_user, proxy_pass)

target_url = 'https://raw.githubusercontent.com/rrenaud/Gibberish-Detector/master/big.txt'
# response = requests.get(target_url, proxies = {'http' : "https://%s:%s@www-proxy-west.aero.org:8080"%(proxy_user, proxy_pass), 'https' : "https://%s:%s@www-proxy-west.aero.org:8080"%(proxy_user, proxy_pass)})
# data = response.text
file = req.urlopen(target_url)
data = ' '.join([line.decode('utf-8') for line in file])

Detector = detector.Detector(trainer.train_on_content(data, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'), threshold=4.0)

import fitz
import datefinder
import subprocess
import datetime
import numpy as np
import re
import glob
import pandas as pd
import hashlib
import json
import sys
import multiprocessing as mp
from collections import Counter
import nltk
#nltk.download('punkt')
import string
import gensim
import spacy
spacy.load('en_core_web_lg')
import en_core_web_md
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer, SnowballStemmer 
from nltk.corpus import wordnet
from tqdm import tqdm
import inflect 
import swifter
import regex
from arxiv_download import get_arxiv_entities

from collections import Counter
import math
from mongodb import MongoDB
from langdetect import detect_langs
from language_parsers.russian_parser.russian_to_english import ru_to_en
from language_parsers.chinese_parser.chinese_to_english import zh_to_en
# For normalize 
# for dependency in ("brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset"):
#     nltk.download(dependency)

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

def cleanup(df, col='Text', remove_meta_data=True, remove_punct=True, add_acronym_periods=False, lemmatize=True, replace_math=True, 
                            remove_non_alpha=True, replace_numbers=True, check_pos=True, remove_SW=True, remove_gibberish=True):
    """
    Clean text according to input parameters
    Input: df with intended text in "Text" column, as space-delimited string of words (Standard sentence)
    """
    # df.Text stays as string through function until lemmatized 
    #print("Input:")
    #print(df.Text.head())
    
    df[col] = df[col].apply(lambda x: x.replace('-\n', ''))
    df[col] = df[col].apply(lambda x: x.replace('\n', ' '))
    
    if replace_math:
        print('removing math symbols and letters...')
        df[col] = df[col].apply(lambda z: ' '.join(['' if regex.search(r'\p{Greek}|\p{S}', word) else word for word in z.split()]))
        print('math symbols and letters removed.\n')
        
    if replace_numbers:
        print('removing stand allow digits (or figrue table numbers)...')
        #p = inflect.engine()
        #df[col] = df[col].apply(lambda x: ' '.join([re.sub(r'[−–-]\d', '', word) for word in x.split()]))
        df[col] = df[col].apply(lambda z: ' '.join(['' if re.search(r'\d', word) else word for word in z.split()]))
        #df[col] = df[col].apply(lambda z: ' '.join(['' if word.isdigit() or re.search(r'([0-9]){2,}[a-z]([0-9]){1,}', word) else word for word in z.split()]))
        
        print('stand alone digits removed.\n')
    
    if check_pos:
        print('keep only nouns...')
#         tag_dict = {"J": wordnet.ADJ,
#                     "N": wordnet.NOUN,
#                     "V": wordnet.VERB,
#                     "R": wordnet.ADV}
        # Nouns only
        tag_dict = {'N': wordnet.NOUN}

        df[col] = df[col].swifter.apply(lambda z: ' '.join([word if tag_dict.get(nltk.pos_tag([word])[0][1][0].upper(), False) else '' for word in z.split()]))
        print('nouns kept; everything else removed\n')
    
    if remove_meta_data:
        print('removing meta data...(Names, Dates, Places...)')
        nlp = en_core_web_md.load()
        df[col] = df[col].swifter.apply(lambda x: removing_meta_data(nlp, x)[0])
        print('removed meta data.\n')
        
    if remove_punct:
        # Remove punctuation
        print('removing punctuation...')
        df[col] = df[col].apply(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).replace('  ', ' ').strip())
        print('punctuation removed.\n')
        
    if add_acronym_periods:
        # Add periods to any capital letter sequence since thats probably an acronym (This is for normalization package)
        print('adding acronym periods...')
        df[col] = df[col].apply(lambda s: ' '.join([''.join([c + "." if c.isalpha() else c for c in w])
                                                              if w.isupper()
                                                              else w
                                                              for w in s.split()]))
        print('acronym periods added.\n')
        
        
    if remove_non_alpha:
        print('removing words with chars not a-zA-Z0-9...')
        df[col] = df[col].apply(lambda x: ' '.join(['' if re.search(r'[^a-zA-Z\d\s]', word) else word for word in x.split()]))
        print('removed words with chars not a-zA-Z0-9.\n')
        

    # Lowercase 
    print('making lower-case...')
    df[col] = df[col].map(lambda x: x.lower())
    print("lower-cased.\n")
    
    
    # It likes the stopwords to be lowercase before removing them
    
    if remove_SW:
        # Remove le stop words
        print('removing stop words...')
        nltk_stop_words = nltk.corpus.stopwords.words('english')
        nltk_stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'table', 'figure', 'arxiv', 'sin', 'cos', 'tan', 'log', 'fx', 'ft', 'dx', 'dt', 'xt'])
        # Tokenized text with standard stopwords and punct removed
        df[col] = df[col].swifter.apply(lambda z: ' '.join([t for t in word_tokenize(z)
                                                          if t not in nltk_stop_words]))
        print('stop words removed.\n')
        
    if lemmatize:
        # Lemmatize
        print('lemmitizing...')
        lemmatizer=WordNetLemmatizer()
        df[col] = df[col].swifter.apply(lambda z: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in z.split()]))
        print("lemmatized.\n")

    
    if remove_gibberish:
        print('removing gibberish...')
        
        df[col] = df[col].apply(lambda z: ' '.join(['' if Detector.is_gibberish(word) else word for word in z.split()]))
        print('gibberish removed.\n')
    
    print('remove single letters or super large words (so big they don\'t make sense)...')
    df[col] = df[col].apply(lambda z: ' '.join([word for word in z.split() if len(word) < 20 and len(word) > 1]))
    print('removed single letters and super large words.\n')
    print('done cleaning.\n')
    return df 


def remove_most_frequent_words(top_n, list_text):
    '''
    Removes the most frequent words from the corpus.

    Parameters
    ----------
    top_n (int) - number of words to be removed
    list_text (list of str) - list of texts representing the corpus
    
    Returns
    -------
    list of str, list of str - returns the list of texts from "list_text" and the list of the words removed.
    '''
    print('removing top %s most frequent words...'%top_n)
    top_words = Counter(" ".join(list_text).split()).most_common(top_n)
    print(top_words)
    for i in range(len(list_text)):
        for top_word in top_words:
            list_text[i] = re.sub(r'\s%s\s'%top_word[0], ' ', list_text[i]).replace('  ', ' ')
    print('top words removed.\n')
    return list_text, top_words

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
    return list_text, stem_dict

def calculate_idf(texts):
    '''
    Calculates the IDF score of a corpus

    Parameters
    ----------
    texts (list of str) - the list of texts that represent the corpus

    Returns
    -------
    tuple - a tuple of the idf score and the term associated with the score
    '''
    N = len(texts)
    tD = Counter()
    for d in texts:
        features = d.split()
        for f in features:
            tD[f] += 1
    IDF = []
    for (term,term_frequency) in tD.items():
        term_IDF = math.log(float(N) / term_frequency)
        #term_IDF = term_frequency / float(N)
        IDF.append(( term_IDF, term ))
    IDF.sort(reverse=True)
    return IDF

def get_terms_idf_terms_to_remove(list_text, idf_thresh=-1.5):
    '''
    Get the terms to remove from idf score.

    Parameters
    ----------
    list_text (list of str) - the list of texts that represent the corpus
    idf_thresh (float, optional) - the threshold for acceptable idf scores to keep; lower than threshold will return list to remove (default = -1.5)
    
    Returns
    -------
    list - the list of terms to remove
    '''
    print('remove idf terms less than', idf_thresh)
    terms_to_remove = []
    for (IDF, term) in calculate_idf(list_text):
        if IDF < idf_thresh:
            terms_to_remove.append((IDF, term))
    return terms_to_remove

def remove_words(list_text, terms_to_remove):
    '''
    Removes words from the corpus.

    Parameters
    ----------
    list_text (list of str) - the list of texts that represent the corpus
    terms_to_remove (list) - the list of words that are to be removed from the corpus

    Returns
    -------
    list of str, list - returns the list of texts with terms removed from terms_to_remove list and also the terms_to_remove list as well 
    '''
    print('removing terms...')
    for i in range(len(list_text)):
        for term in terms_to_remove:
            list_text[i] = re.sub(r'\s%s\s'%term[1], ' ', list_text[i]).replace('  ', ' ')
    print('terms removed.\n')
    return list_text, terms_to_remove
    

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
        
#         if i == 2:
#             ngram = gensim.models.Phrases(list_text, min_count=min_count, threshold=threshold)
#             ngram_mod = gensim.models.phrases.Phraser(ngram)
#             texts = [ngram_mod[doc] for doc in texts]
#         else:
#             ngram = gensim.models.Phrases(init_gram[i-1]['ngram'][text], threshold=threshold)
#             ngram_mod = gensim.models.phrases.Phraser(ngram)
#             texts = [ngram_mod[init_gram[i-1]['ngram_mod'][doc]] for doc in texts]
            
        try:
            print(numbers[i], 'model finished.\n')
        except:
            print('%s-gram model finished.\n')
            
        init_gram[i] = {'ngram':ngram, 'ngram_mod':ngram_mod, 'texts':texts}
    
    return [' '.join(text) for text in texts]

def get_sha256(content):
    '''
    Gets the sha256 code from the content of the entry
    
    Args:
        content : the end of week entry content, specifically a string
        
    Returns:
        string : sha256 code
    '''
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

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
                    'date' : date}
    if os.path.join(directory, filename.replace('.pdf', '')+'_entities.json') in glob.glob(os.path.join(directory, '*.json')):
        with open(os.path.join(directory, filename.replace('.pdf', '')+'_entities.json'), 'r') as f:
            arxiv_entities = json.load(f)
        return {**section_dict, **arxiv_entities}
    elif re.search(r'(\d){4}\.(\d){5}', filename):
        try:
            arxiv_entities = get_arxiv_entities(re.search(r'(\d){4}\.(\d){5}', filename)[0])
            return {**section_dict, **arxiv_entities}
        except:
            pass
    return section_dict

def parse_clean_pdf(directory, filename, stem=False, abstract=True, idf_thresh=-1.5):
    '''
    Parses and cleans a pdf file by extracting document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the document
    filename (str) - the name of the pdf file
    stem (bool, optional) - True means to stem the words and False is to leave as is (default = False)
    abstract (bool, optional) - True means to normalize the abstract (default = False)
    idf_thresh (float, optional) - the threshold for acceptable idf scores to keep; lower than threshold will return list to remove (default = -1.5)

    Returns
    -------
    dict - a dictionary with the following keys: Document, Abstract, Text, Abstract_Original, Original_Text, Path, sha_256, laguage, language_probability, Authors, Tilte, url, date
    '''
    filepath = os.path.join(directory,filename)
    if filepath.endswith('.pdf'):
        pdf_dict = pdf_to_dict(directory, filename)
        if pdf_dict == {}:
            return {}
        df = pd.DataFrame([pdf_dict])
        df = cleanup(df, col='Text', remove_meta_data=True, remove_punct=True, add_acronym_periods=False, lemmatize=True, replace_math=True, 
                            remove_non_alpha=True, replace_numbers=True, check_pos=True, remove_SW=True, remove_gibberish=True)
        if stem:
            df.Text, _ = stem_words(df.Text)
        if idf_thresh is not None:
            idf_terms_to_remove = get_terms_idf_terms_to_remove(df.Text, idf_thresh=idf_thresh)
            df.Text, idf_terms_to_remove = remove_words(df.Text, idf_terms_to_remove)

        if abstract:
            df = cleanup(df, col='Abstract', remove_meta_data=True, remove_punct=True, add_acronym_periods=False, lemmatize=True, replace_math=True, 
                                remove_non_alpha=True, replace_numbers=True, check_pos=True, remove_SW=True, remove_gibberish=True)
            if stem:
                df.Abstract, _ = stem_words(df.Abstract)
            if idf_thresh is not None:
                df.Abstract, _ = remove_words(df.Abstract, idf_terms_to_remove)
        
        return df.T.to_dict().values()
    else:
        return None

def parse_clean_pdfs(directory, for_training=True, n_most_freq_words_remove=10, stem=False, abstract=True, idf_thresh=-1.5):#, storage_path=None):
    '''
    Parses and cleans a directory of pdfs by extracting from each: document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the pdf documents
    for_training (bool, optional) - True if this is parse and cleaning is done for training models, False if this is used for testing or production (default = True)
    n_most_freq_words_remove (int, optional) - the number of most popular words to be removed from the corpus, which will only run if for_training = True (default = 10)
    stem (bool, optional) - True means to stem the words and False is to leave as is (default = False)
    abstract (bool, optional) - True means to normalize the abstract (default = False)
    idf_thresh (float, optional) - the threshold for acceptable idf scores to keep; lower than threshold will return list to remove (default = -1.5)

    Returns
    -------
    list of dicts - a list of dictionary with the following keys: Document, Abstract, Text, Abstract_Original, Original_Text, Path, sha_256, laguage, language_probability, Authors, Tilte, url, date
    '''
    
    all_docx = os.listdir(directory)
    #all_docx.sort()
    #print(all_docx)
    
    docs = []
    
    
    for filepath in all_docx:
        if filepath.endswith('.pdf'):
            print(os.path.join(directory,filepath))
            pdf_dict = pdf_to_dict(directory, filepath)
            if pdf_dict == {}:
                continue
            docs.append(pdf_dict)
            
    df = pd.DataFrame(docs)
    print('check point: df made')
    print('processing text...')
    df = cleanup(df, col='Text', remove_meta_data=True, remove_punct=True, add_acronym_periods=False, lemmatize=True, replace_math=True, 
                            remove_non_alpha=True, replace_numbers=True, check_pos=True, remove_SW=True, remove_gibberish=True)
    if stem:
        df.Text, stem_dict = stem_words(df.Text)
    if idf_thresh is not None:
        idf_terms_to_remove = get_terms_idf_terms_to_remove(df.Text, idf_thresh=idf_thresh)
        df.Text, idf_terms_to_remove = remove_words(df.Text, idf_terms_to_remove)
    df.Text = ngram(df.Text, n=3)
    print('processed text.\n')
    
    if abstract:
        print('processing abstract...')
        df = cleanup(df, col='Abstract', remove_meta_data=True, remove_punct=True, add_acronym_periods=False, lemmatize=True, replace_math=True, 
                            remove_non_alpha=True, replace_numbers=True, check_pos=True, remove_SW=True, remove_gibberish=True)
        if stem:
            df.Abstract, stem_dict = stem_words(df.Abstract)
        if idf_thresh is not None:
                df.Abstract, _ = remove_words(df.Abstract, idf_terms_to_remove)
        df.Abstract = ngram(df.Abstract, n=3)
        print('processed abstract.\n')
        
    if for_training:
        if isinstance(n_most_freq_words_remove, int) and n_most_freq_words_remove > 0:
            df.Text, removed_words = remove_most_frequent_words(n_most_freq_words_remove, df.Text)
            print('removed words:', removed_words)
            with open(os.path.join(directory, 'removed_top_n_words.json'), 'w') as j_file:
                json.dump(removed_words, j_file, indent=4)
        if stem:
            with open(os.path.join(directory, 'stem_dict.json'), 'w') as j_file:
                json.dump(stem_dict, j_file, indent=4)
            
    return df.T.to_dict().values()

def parsed_pdf_to_dict(directory, for_training=True, n_most_freq_words_remove=10, stem=False, abstract=True, idf_thresh=-1.5):
    '''
    Parses and cleans a directory of pdfs by extracting from each: document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the pdf documents
    for_training (bool, optional) - True if this is parse and cleaning is done for training models, False if this is used for testing or production (default = True)
    n_most_freq_words_remove (int, optional) - the number of most popular words to be removed from the corpus, which will only run if for_training = True (default = 10)
    stem (bool, optional) - True means to stem the words and False is to leave as is (default = False)
    abstract (bool, optional) - True means to normalize the abstract (default = False)
    idf_thresh (float, optional) - the threshold for acceptable idf scores to keep; lower than threshold will return list to remove (default = -1.5)

    Returns
    -------
    dict of dicts - a dictionary with the each sha256 hash of the orginal text as the key of each dictionary with the following keys: Document, Abstract, Text, Abstract_Original, Original_Text, Path, sha_256, laguage, language_probability, Authors, Tilte, url, date
    '''
    if directory.endswith('.pdf'):
        parsed_list = [parse_clean_pdf(directory='.', filename=directory, stem=stem, abstract=abstract, idf_thresh=idf_thresh)]
    else:
        parsed_list = parse_clean_pdfs(directory, for_training, n_most_freq_words_remove, stem=stem, abstract=abstract, idf_thresh=idf_thresh)
    
    return {doc['sha_256'] : doc for doc in parsed_list}

def parsed_pdf_to_json(directory, storage_dir='./parsed_cleaned_pdfs', for_training=True, n_most_freq_words_remove=10, stem=False, abstract=True, idf_thresh=-1.5):
    '''
    Parses and cleans and stores jsons of a directory of pdfs by extracting from each: document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the pdf documents
    storage_dir (str, optional) - the directory that the jsons of parsed documents will be stored
    for_training (bool, optional) - True if this is parse and cleaning is done for training models, False if this is used for testing or production (default = True)
    n_most_freq_words_remove (int, optional) - the number of most popular words to be removed from the corpus, which will only run if for_training = True (default = 10)
    stem (bool, optional) - True means to stem the words and False is to leave as is (default = False)
    abstract (bool, optional) - True means to normalize the abstract (default = False)
    idf_thresh (float, optional) - the threshold for acceptable idf scores to keep; lower than threshold will return list to remove (default = -1.5)

    '''
    
    try:
        os.mkdir(storage_dir)
    except (FileExistsError):
        pass
        
    parsed_dict = parsed_pdf_to_dict(directory, for_training, n_most_freq_words_remove, stem=stem, abstract=abstract, idf_thresh=idf_thresh)
    
    for doc in parsed_dict.values():
        with open(os.path.join(storage_dir, str(doc['sha_256'])+'.json'), 'w') as j_file:
                json.dump(doc, j_file, indent=4)

def parsed_pdf_to_db(directory, mongoclient_dict, database, collection, for_training=True, n_most_freq_words_remove=10, stem=False, abstract=True, idf_thresh=-1.5):
    '''
    Parses and cleans and stores dictionaries in mongoDB database of a directory of pdfs by extracting from each: document name, abstract, normalized text, original text, normalized abstract, path, sha256 hash, language, language probability, and date (if arxiv document authors, title, and url as well).

    Parameters
    ----------
    directory (str) - directory that contains the pdf documents
    mongoclient_dict (dict) - the mongoDB client dictionary containing exceptable inputs to pymongo.MongoClient class.  Examples are host, username, password, authSource, etc. 
    database (str) - the database in the mongoDB host
    collection (str) - the collection in the database to populate
    for_training (bool, optional) - True if this is parse and cleaning is done for training models, False if this is used for testing or production (default = True)
    n_most_freq_words_remove (int, optional) - the number of most popular words to be removed from the corpus, which will only run if for_training = True (default = 10)
    stem (bool, optional) - True means to stem the words and False is to leave as is (default = False)
    abstract (bool, optional) - True means to normalize the abstract (default = False)
    idf_thresh (float, optional) - the threshold for acceptable idf scores to keep; lower than threshold will return list to remove (default = -1.5)

    '''
    parsed_dict = parsed_pdf_to_dict(directory, for_training, n_most_freq_words_remove, stem=stem, abstract=abstract, idf_thresh=idf_thresh)

    mongo = MongoDB(mongoclient_dict, database, collection)
    mongo.update_many_documents(parsed_dict)
                
if __name__ == '__main__':
    directory = './math_pdfs'
    storage_dir = './parsed_cleaned_pdfs'

    parsed_pdf_to_json(directory, storage_dir, for_training=True, n_most_freq_words_remove=0, stem=False)