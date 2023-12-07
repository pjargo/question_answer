import torch
from gensim.models import Word2Vec
import os
import re
import nltk
import spacy
import numpy as np
from urllib.parse import quote_plus
from .utils import remove_non_word_chars, clean_text, tokens_to_embeddings, post_process_output, correct_spelling, \
    timing_decorator
from .mongodb import MongoDb
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForQuestionAnswering, RobertaTokenizer, RobertaForQuestionAnswering
import time
from .config import TOKENIZER, EMBEDDING_MODEL_FNAME, EMBEDDING_MODEL_TYPE, TOKENS_EMBEDDINGS, database_name, \
    DOCUMENT_TOKENS, TOP_N, TRANSFORMER_MODEL_NAME, METHOD, MAX_QUERY_LENGTH, username, password, cluster_url, \
    mongo_host, mongo_port, mongo_username, mongo_password, mongo_auth_db, mongo_database_name
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import platform

# Set proxy information if windows
if platform.system() == "Windows":
    print("Running on Windows")
    # Get the current date and time
    now = datetime.now()
    day = now.strftime("%A")
    proxy_url = f"http://33566:{day[0:3]}@proxy-west.aero.org:8080"

    # Set proxy environment variables
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
else:
    print("Running on macOS!")


class QuestionAnswer:
    def __init__(self):
        # Set the Tokenizer for your specific BERT model variant
        bert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        roberta_tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2", add_prefix_space=True)
        tokenizers = {'bert': bert_base_tokenizer, 'roberta': roberta_tokenizer}
        self.tokenizer = tokenizers[TOKENIZER]

        # Load your trained Word2Vec model
        if EMBEDDING_MODEL_TYPE == 'Word2Vec':
            self.embedding_model = Word2Vec.load(
                os.path.join(os.getcwd(), "question_answer", "embedding_models", EMBEDDING_MODEL_FNAME))
        elif EMBEDDING_MODEL_TYPE.lower() == 'glove':
            # Load the custom spaCy model
            self.embedding_model = spacy.load(os.path.join(os.getcwd(), "question_answer", "embedding_models",
                                                           EMBEDDING_MODEL_FNAME.split(".bin")[0]))

        # BERT or ROBERTA model?
        if TRANSFORMER_MODEL_NAME.lower() in ['bert', 'bert-base-uncased', 'bert_base']:
            # Load the pre-trained BERT model and tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
            self.transformer_model = BertForQuestionAnswering.from_pretrained(TRANSFORMER_MODEL_NAME)
        else:
            # Load the pre-trained RoBERTa model and tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME, add_prefix_space=True)
            self.transformer_model = RobertaForQuestionAnswering.from_pretrained(TRANSFORMER_MODEL_NAME)

        # Specify Candidate token embeddings option
        if TOKENS_EMBEDDINGS == "query":
            self.TOKENS, self.EMBEDDINGS = "tokenized_query", "query_embedding"
        elif TOKENS_EMBEDDINGS == "query_search":
            self.TOKENS, self.EMBEDDINGS = "tokenized_query_search", "query_embedding_search"
        else:
            self.TOKENS, self.EMBEDDINGS = "tokenized_query_search_less_sw", "query_embedding_search_less_sw"

        # Escape the username and password
        self.escaped_username = quote_plus(username)
        self.escaped_password = quote_plus(password)

        # Escape the username and password for Aerospace Mongo Credentials
        self.aero_escaped_username = quote_plus(mongo_username)
        self.aero_escaped_password = quote_plus(mongo_password)

    @timing_decorator("Total execution time")
    def answer_question(self, query: str):
        """
        Find answer in corpus of documents
        :param query: (str) User input question
        :return: JSON containing answer, source data and recommended documents if no answer found
        """
        query_data = self.process_query(query)

        # Get the candidate documents, top_n_documents: (similarity_score, document dictionary)
        top_n_documents = self.get_candidate_docs(query_data)
        top_n_documents.sort(key=lambda x: x[1]['counter'])

        # Get answer with possible answers (list of dictionaries {confidence: , doc:{} })
        answers = self.get_answers(query_data, top_n_documents)

        # Fetch the source document text
        source_text_dict, doc_rec_set = None, None
        if answers:
            source_text_dict = self.fetch_source_documents(answers)
        else:
            doc_rec_set = set([doc_info[1]['Document'] for doc_info in top_n_documents])

        return {"query": query_data["query"], "results": answers,
                "source_text_dictionary": source_text_dict, 'no_ans_found': doc_rec_set}

    @timing_decorator("Execution time to fetch source documents")
    def fetch_source_documents(self, detected_answers):
        """
        Get all documents from _extracted_text collection, For highlighting and displaying answers

        :param detected_answers: [[dict]] a list of dictionaries containing the answers to the query
        :return: (dict) key -> source document name, value -> Full text for document from Mongo
        """
        # Get the list of unique documents
        unique_documents = set()
        for ans_dict in detected_answers:
            unique_documents.add(ans_dict['document'])

        if platform.system() == "Darwin":
            # Personal Mongo instance
            mongodb = MongoDb(username=self.escaped_username,
                              password=self.escaped_password,
                              cluster_url=cluster_url,
                              database_name=database_name,
                              collection_name="extracted_text")
        else:
            # Aerospace credentials
            mongodb = MongoDb(username=self.aero_escaped_username,
                              password=self.aero_escaped_password,
                              database_name=mongo_database_name,
                              mongo_host=mongo_host,
                              collection_name="extracted_text",
                              mongo_port=mongo_port,
                              mongo_auth_db=mongo_auth_db)

        source_text_dict = dict()
        if mongodb.connect():
            for source_doc_name in unique_documents:
                source_text_dict[source_doc_name] = \
                    list(mongodb.get_collection().find({'Document': source_doc_name}))[0]['Text']

        return source_text_dict

    @timing_decorator("Execution time to get answers from transformer")
    def get_answers(self, query_data, candidate_documents):
        """
        Inspect top N documents for answer based on input query

        :param query_data: (dict)
        :param candidate_documents: ( [(float, dict)] ) -> (similarity_score, document dictionary)
        :return: [dict] keys -> "confidence_score", "answer", "context", "document"
        """
        # Concatenate tokens from all candidate chunks
        candidate_docs_tokens_concatenated = []
        prev_doc = None
        for sim_score, candidate in candidate_documents:
            candidate_docs_tokens = candidate["tokens"]

            # Add a separation token between documents unless they are consecutive numbers, indicating they are not
            # separate chunks
            if prev_doc and int(candidate['counter']) != int(prev_doc['counter']) + 1:
                candidate_docs_tokens_concatenated.extend([self.tokenizer.sep_token_id])

            candidate_docs_tokens_concatenated.extend(candidate_docs_tokens)
            prev_doc = candidate

        chunks_info = [(candidate['tokens'], candidate['Document'], candidate['counter']) for sim_score, candidate in
                       candidate_documents]
        # Process each chunk separately and store logits
        with torch.no_grad():
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                # Submit each chunk for processing concurrently
                futures = [executor.submit(self.get_answer_and_confidence, chunk, doc, cntr, query_data) for
                           (chunk, doc, cntr) in chunks_info]

                # Wait for all tasks to complete
                candidate_responses_list = [future.result() for future in futures if
                                            "Sorry, I don't have information on that topic." not in future.result().get(
                                                "answer", "")]

        return candidate_responses_list

    def get_answer_and_confidence(self, chunk, document, counter, query):
        """
        Get answer from transformer from query and chunk as context

        :param counter: (int)
        :param chunk:([str]) list of tokens as strings, document from mongo databse
        :param document: (str) Document the chunk is derived from
        :param query: (dict) Processed query
        :return: (dict) keys:values
            -> "confidence_score": (float) confidence of the output
            -> "answer": (str) Representing the answer from the transformer output logits
            -> "context": (str) Context input into transformer model
            -> "document": (str) Source document
        """
        chunk = [str(token) if isinstance(token, int) else token for token in chunk]
        inputs = self.tokenizer.encode_plus(query["tokenized_query"], chunk, max_length=512, return_tensors="pt",
                                            padding="max_length", truncation=True)

        # Roberta Model does not have token_type_ids as an input argument
        if TRANSFORMER_MODEL_NAME.lower() in ['bert', 'bert-base-uncased', 'bert_base']:
            outputs = self.transformer_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                             token_type_ids=inputs["token_type_ids"])
        else:
            outputs = self.transformer_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        combined_scores = start_logits.unsqueeze(-1) + end_logits.unsqueeze(1)

        # Find the indices with the highest combined score
        max_combined_score_idx = torch.argmax(combined_scores)

        # Convert the index to start and end indices
        start_idx = torch.div(max_combined_score_idx, combined_scores.size(1), rounding_mode='trunc')
        end_idx = max_combined_score_idx - start_idx * combined_scores.size(1)

        answer_tokens = inputs["input_ids"][0][start_idx: end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answer = post_process_output(answer)

        if answer == "":
            answer = "Sorry, I don't have information on that topic."
        else:
            print(f"Answer found in chunk count: {counter}")
        if query['query'] in answer:  # If roberta return logits are entire context
            escaped_query = re.escape(query['query'])
            answer = re.sub(escaped_query, "", answer)

        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)
        confidence_score = start_probs[0, start_idx] * end_probs[0, end_idx]

        context = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

        return {"confidence_score": confidence_score.item(), "answer": answer, "context": context, "document": document}

    @timing_decorator(f"Execution time to retrieve top {TOP_N} candidate documents")
    def get_candidate_docs(self, query_data):
        """
        Get similarity score between query embeddings and all document embeddings, sort by score and return top N

        :param query_data: (dict) Processed query
        :return: [(float, dict)] sorted list of tuples containing similarity score and data from Mongo
        """
        documents = self.get_documents_from_mongo()
        with ThreadPoolExecutor() as executor:
            # Submit each document for processing concurrently
            futures = [executor.submit(self.get_doc_sim_scores, doc, query_data) for doc in documents]

            # Wait for all tasks to complete
            sim_scores = [future.result() for future in futures]
        sim_scores.sort(key=lambda x: x[0], reverse=True)

        return sim_scores[:TOP_N]

    def get_doc_sim_scores(self, document, query_data):
        """
        Cosine similarity score between query embedding and
            - BOTH: combination of COMBINE_MEAN and MEAN_MAX
            - MEAN_MAX: Avg. maximum cosine similarity between each embedding in query and embeddings in chunk
            - COMBINE_MEAN: Cosine similarity between avg. embedding in query and avg. embedding in chunk
            - MEAN_MEAN: Avg. Cosine similarity between embedding in query and embedding in chunk

        :param document: (dict) Mongo queried document from parsed_documents
        :param query_data: (dict) Processed query
        :return: (float, dict) Similarity score and original document data
        """
        query_embedding = np.array(query_data[self.EMBEDDINGS])
        query_tokens = np.array(query_data[self.TOKENS])

        # remove the paddings from the query
        query_embedding = np.array([emb for emb, token in zip(query_embedding, query_tokens) if token != '[PAD]'])

        # List to store cosine similarity scores and corresponding document filenames
        chunk_tokens = np.array(document[DOCUMENT_TOKENS])
        chunk_embeddings = tokens_to_embeddings(document[DOCUMENT_TOKENS], self.embedding_model)

        # remove the paddings and unknown tokens from the query
        chunk_embeddings = np.array(
            [emb for emb, token in zip(chunk_embeddings, chunk_tokens) if token not in ['[PAD]', '[UNK]']])

        # Calculate cosine similarity between query_embedding and chunk_embeddings METHOD = 'MEAN_MAX'
        if METHOD == 'MEAN_MAX':
            similarity = cosine_similarity(query_embedding, chunk_embeddings)
            similarity = np.mean(np.max(similarity, axis=1))
        elif METHOD == 'MEAN_MEAN':
            similarity = cosine_similarity(query_embedding, chunk_embeddings)
            similarity = np.mean(similarity)
        elif METHOD == 'COMBINE_MEAN':  # 'COMBINE_MEAN'
            similarity = cosine_similarity(np.mean(query_embedding, axis=0).reshape(1, -1),
                                           np.mean(chunk_embeddings, axis=0).reshape(1, -1))
            similarity = np.mean(similarity)  # Get the single value out of the array
        else:
            mean_max_similarity = cosine_similarity(query_embedding, chunk_embeddings)
            mean_max_similarity = np.mean(np.max(mean_max_similarity, axis=1))
            combine_mean_similarity = cosine_similarity(np.mean(query_embedding, axis=0).reshape(1, -1),
                                                        np.mean(chunk_embeddings, axis=0).reshape(1, -1))
            combine_mean_similarity = np.mean(combine_mean_similarity)
            similarity = .5 * mean_max_similarity + .5 * combine_mean_similarity

        return (similarity, document)

    @timing_decorator("Time to fetch all documents")
    def get_documents_from_mongo(self):
        """
        Get all documents from the parsed_documents collection in Mongodb

        :return: [dict] Keys -> tokens, tokens_less_sw, counter, Document
        """
        if platform.system() == "Darwin":
            # Personal Mongo instance
            mongodb = MongoDb(username=self.escaped_username,
                              password=self.escaped_password,
                              cluster_url=cluster_url,
                              database_name=database_name,
                              collection_name="parsed_documents")
        else:
            # Aerospace credentials
            mongodb = MongoDb(username=self.aero_escaped_username,
                              password=self.aero_escaped_password,
                              database_name=mongo_database_name,
                              mongo_host=mongo_host,
                              collection_name="parsed_documents",
                              mongo_port=mongo_port,
                              mongo_auth_db=mongo_auth_db)

        if mongodb.connect():
            documents = mongodb.get_documents(query={}, inclusion={"tokens": 1, "tokens_less_sw": 1, "counter": 1,
                                                                   "Document": 1, "_id": 0})
            documents = list(documents)
            print(f"Total documents: {mongodb.count_documents()}")
            mongodb.disconnect()
            return documents
        return []

    @timing_decorator("Execution time to process query")
    def process_query(self, user_query: str):
        """
        Prepare query from similarity score and transformer

        :param user_query: (str)
        :return: dict
        """
        user_query = user_query.lower()

        # clean query for BERT input
        user_query = clean_text(user_query)
        print("Uncorrected query: ", user_query)
        user_query = self.spell_check(user_query)
        print("Corrected query: ", user_query)

        # clean query for candidate search
        user_query_for_search = remove_non_word_chars(user_query)

        # Tokenize the query for BERT input
        tokenized_query = self.tokenizer.tokenize(user_query)

        # Tokenize the query for candidate search
        tokenized_query_for_search = self.tokenizer.tokenize(user_query_for_search)

        # Remove the stop words for the tokenized query for search
        nltk_stop_words = nltk.corpus.stopwords.words('english')
        nltk_stop_words.extend(["Ġ" + word for word in nltk_stop_words])  # Add the roberta modified tokens
        tokenized_query_for_search_less_sw = [token for token in tokenized_query_for_search if
                                              token not in nltk_stop_words]

        # Pad or truncate the query to a fixed length of 20 tokens (BERT input)
        if len(tokenized_query) > MAX_QUERY_LENGTH:
            tokenized_query = tokenized_query[:MAX_QUERY_LENGTH]
        else:
            padding_length = MAX_QUERY_LENGTH - len(tokenized_query)
            tokenized_query = tokenized_query + [self.tokenizer.pad_token] * padding_length

        # Convert the tokenized query to input IDs and attention mask
        input_ids_query = self.tokenizer.convert_tokens_to_ids(tokenized_query)
        attention_mask_query = [1] * len(input_ids_query)

        # Convert to tensors
        input_ids_query = torch.tensor(input_ids_query).unsqueeze(0)  # Add batch dimension
        attention_mask_query = torch.tensor(attention_mask_query).unsqueeze(0)  # Add batch dimension

        # Get the query embeddings for the candidate document search
        query_embeddings = tokens_to_embeddings(tokenized_query, self.embedding_model, RANDOM=False)
        query_embeddings_search = tokens_to_embeddings(tokenized_query_for_search, self.embedding_model, RANDOM=False)
        query_embeddings_less_sw = tokens_to_embeddings(tokenized_query_for_search_less_sw, self.embedding_model,
                                                        RANDOM=False)

        query_data = {
            "query": user_query,
            "input_ids_query": input_ids_query.tolist(),
            "attention_mask_query": attention_mask_query.tolist(),
            "query_search": user_query_for_search,
            "tokenized_query": tokenized_query,
            "tokenized_query_search": tokenized_query_for_search,
            "tokenized_query_search_less_sw": tokenized_query_for_search_less_sw,
            "query_embedding": query_embeddings,  # Just used for the candidate search
            "query_embedding_search": query_embeddings_search,  # Just used for the candidate search, cleaned
            "query_embedding_search_less_sw": query_embeddings_less_sw  # .tolist()
        }
        return query_data

    def spell_check(self, user_query) -> str:
        """
        Check from spelling errors in query
         - Tokenize query
         - Construct the words from the tokenized query
         - For each word, ensure all tokens are in the word embedding model
         - Otherwise spell check word

        :param user_query:
        :return: (str)
        """
        tokenized_query = self.tokenizer.tokenize(user_query)

        # Group tokens into words
        words = []
        current_word = ""
        for token in tokenized_query:
            if token.startswith("Ġ"):  # Indicates the start of a new word
                if current_word:
                    words.append(current_word)
                current_word = token[1:] if token[1:] not in ['(', '[', '{', '/', '\\'] else ''
            else:
                current_word += token if token not in [')', ']', '}', '/', '\\', '?', ".", "!"] else ''
                if token in ['/', '\\']:
                    words.append(current_word)
                    current_word = ''
        if current_word:
            words.append(current_word)

        # Identify misspelled words not in the embeddings model
        misspelled_words = []
        for word in words:
            # Split punctuation and hyphens from the word
            base_word = "".join(char for char in word if char.isalnum() or char in ["'", "-"])
            if any(list(map(lambda x: not any(x),
                            tokens_to_embeddings(self.tokenizer.tokenize(base_word), self.embedding_model,
                                                 RANDOM=False)))):
                # Add the original word to the misspelled_words list
                misspelled_words.append(word)
        # Correct the spelling of misspelled words
        corrected_words = {word: correct_spelling(word) for word in misspelled_words}

        # Replace misspelled words in the original query
        corrected_query = user_query
        for original, corrected in corrected_words.items():
            corrected_query = corrected_query.replace(original, corrected)

        return corrected_query
