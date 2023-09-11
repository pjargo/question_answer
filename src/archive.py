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
    print(i_start, i_end)
    
    return full_text[i_start:i_end], full_text[i_start:i_end_abstract], langs


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