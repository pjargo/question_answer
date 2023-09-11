from transformers import BertTokenizer, RobertaTokenizer
import torch
from gensim.models import Word2Vec
import os
import json
import re
import nltk
import spacy
import numpy as np
from urllib.parse import quote_plus
from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient
from .utils import get_sha256, clean_text, remove_non_word_chars, clean_text, tokens_to_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForQuestionAnswering, RobertaTokenizer, RobertaForQuestionAnswering


hyperparams = {
    "TOKENIZER": "roberta",
    "input_folder": "space_based_pdfs",
    "embedding_model_type": "glove",
    "embedding_model_fname": "roberta_space_based_pdfs_glove_model.bin",
    "vector_size": 50,
    "window": 3,
    "min_count": 3,
    "sg": 0,
    "TOKENS_TPYE": "tokens_less_sw",
    "chunk_size": 350,
    "chunk_overlap": 0,
    "max_query_length": 20,
    "top_N": 5,
    "TOKENS_EMBEDDINGS": "query_search_less_sw",
    "DOCUMENT_EMBEDDING": "token_embeddings_less_sw",
    "DOCUMENT_TOKENS": "tokens_less_sw",
    "METHOD": "MEAN_MAX",
    "transformer_model_name": "deepset/roberta-base-squad2",
    "context_size": 500
}


class MongoDb:
    def __init__(self, username, password, cluster_url, database_name=None, collection_name=None):
        self.username = username
        self.password = password
        self.cluster_url = cluster_url
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None

    def connect(self):
        try:
            uri = f"mongodb+srv://{self.username}:{self.password}@{self.cluster_url}.pog6zw2.mongodb.net/?retryWrites=true&w=majority"
            # Create a new client and connect to the server
            self.client = MongoClient(uri, server_api=ServerApi('1'))
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None

    def get_collection(self):
        if self.client:
            db = self.client[self.database_name]
            collection = db[self.collection_name]
            return collection
        else:
            return None

    def insert_document(self, document):
        collection = self.get_collection()
        if collection is not None:
            try:
                result = collection.insert_one(document)
                return result.inserted_id
            except Exception as e:
                print(f"Error inserting document: {e}")
        return None

    def count_documents(self):
        collection = self.get_collection()
        if collection is not None:
            try:
                count = collection.count_documents({})
                return count
            except Exception as e:
                print(f"Error counting documents: {e}")
        return None

    def iterate_documents(self):
        collection = self.get_collection()
        if collection is not None:
            cursor = collection.find({})
            for document in cursor:
                yield document


class QuestionAnswer():
    def __init__(self):
        # Set the Tokenizer for your specific BERT model variant
        TOKENIZER = hyperparams["TOKENIZER"]
        bert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        roberta_tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2", add_prefix_space=True)
        tokenizers = {'bert': bert_base_tokenizer, 'roberta': roberta_tokenizer}
        self.tokenizer = tokenizers[TOKENIZER]

        # Load your trained Word2Vec model
        embedding_model_fname = hyperparams["embedding_model_fname"]
        embedding_model_type = hyperparams['embedding_model_type']
        if embedding_model_type == 'Word2Vec':
            self.model = Word2Vec.load(
                os.path.join(os.getcwd(), "question_answer", "embedding_models", embedding_model_fname))
        elif embedding_model_type.lower() == 'glove':
            # Load the custom spaCy model
            self.model = spacy.load(os.path.join(os.getcwd(), "question_answer", "embedding_models",
                                                 embedding_model_fname.split(".bin")[0]))

        # Specify the tokens and embeddings for the query
        TOKENS_EMBEDDINGS = hyperparams['TOKENS_EMBEDDINGS']
        # Specify Candidate token embeddings option
        self.DOCUMENT_EMBEDDING = hyperparams['DOCUMENT_EMBEDDING']
        self.DOCUMENT_TOKENS = hyperparams['DOCUMENT_TOKENS']
        if TOKENS_EMBEDDINGS == "query":
            self.TOKENS = "tokenized_query"
            self.EMBEDDINGS = "query_embedding"
        elif TOKENS_EMBEDDINGS == "query_search":
            self.TOKENS = "tokenized_query_search"
            self.EMBEDDINGS = "query_embedding_search"
        # elif TOKENS_EMBEDDINGS == "query_search_less_sw":
        else:
            self.TOKENS = "tokenized_query_search_less_sw"
            self.EMBEDDINGS = "query_embedding_search_less_sw"

        self.METHOD = hyperparams['METHOD']
        self.top_N = hyperparams['top_N']

        self.model_name = hyperparams["transformer_model_name"]

        self.context_size = hyperparams["context_size"]

        # Set mongoDb information
        self.username = "new_user_1"
        self.password = "password33566"
        self.cluster_url = "cluster0"
        self.database_name = "question_answer"
        self.collection_name = "parsed_documents"

    def answer_question(self, query):
        query_data = self.process_query(query)

        # Get the candidate documents, top_n_documents: (similarity_score, document dictionary)
        top_n_documents = self.get_candidate_docs(query_data)
        top_n_documents.sort(key=lambda x: x[1]['counter'])

        # Get answer with possible answers (list of dictionaries {confidence: , doc:{} })
        answers = self.get_answer(query_data, top_n_documents)
        return {"query": query_data["query"], "results": answers}

    def get_answer(self, query_data, candidate_documents):
        # BERT or ROBERTA model?
        if self.model_name.lower() in ['bert', 'bert-base-uncased', 'bert_base']:
            # Load the pre-trained BERT model and tokenizer
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertForQuestionAnswering.from_pretrained(self.model_name)
        else:
            # Load the pre-trained RoBERTa model and tokenizer
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
            model = RobertaForQuestionAnswering.from_pretrained(self.model_name)

        query_tokens = query_data["tokenized_query"]
        # Concatenate tokens from all candidate chunks
        candidate_docs_tokens_concatenated = []
        prev_doc = None
        for sim_score, candidate in candidate_documents:
            candidate_docs_tokens = candidate["tokens"]

            # Add a separation token between documents unless they are consecutive numbers, indicating they are not
            # separate chunks
            if prev_doc and int(candidate['counter']) != int(prev_doc['counter']) + 1:
                candidate_docs_tokens_concatenated.extend([tokenizer.sep_token_id])

            candidate_docs_tokens_concatenated.extend(candidate_docs_tokens)
            prev_doc = candidate

        chunks = [candidate_docs_tokens_concatenated[i:i + self.context_size] for i in
                  range(0, len(candidate_docs_tokens_concatenated), self.context_size)]

        chunks = [(candidate['tokens'], candidate['Document']) for sim_score, candidate in candidate_documents]
        print(chunks)
        candidate_responses_list = list()
        # Process each chunk separately and store logits
        with torch.no_grad():
            for i, (chunk, doc) in enumerate(chunks):
                chunk = [str(token) if isinstance(token, int) else token for token in chunk]

                inputs = tokenizer.encode_plus(query_tokens, chunk, max_length=512, return_tensors="pt",
                                               padding="max_length", truncation=True)

                # Roberta Model does not have token_type_ids as an input argument
                if self.model_name.lower() in ['bert', 'bert-base-uncased', 'bert_base']:
                    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                    token_type_ids=inputs["token_type_ids"])
                else:
                    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

                answer_confidence_dict = self.get_answer_and_confidence(inputs, outputs, 0)
                answer_confidence_dict["document"] = doc
                candidate_responses_list.append(answer_confidence_dict)

        return candidate_responses_list

    def get_answer_and_confidence(self, input, output, confidence_threshold=0):
        start_logits = output.start_logits
        end_logits = output.end_logits

        combined_scores = start_logits.unsqueeze(-1) + end_logits.unsqueeze(1)

        # Find the indices with the highest combined score
        max_combined_score_idx = torch.argmax(combined_scores)

        # Convert the index to start and end indices
        start_idx = torch.div(max_combined_score_idx, combined_scores.size(1), rounding_mode='trunc')
        end_idx = max_combined_score_idx - start_idx * combined_scores.size(1)
        print(f"Start: {start_idx} \t End: {end_idx}")

        answer_tokens = input["input_ids"][0][start_idx: end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answer = self.post_process_output(answer)
        if answer == "":
            answer = "Sorry, I don't have information on that topic."

        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)
        confidence_score = start_probs[0, start_idx] * end_probs[0, end_idx]

        context = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input["input_ids"][0]))
        print("Context: ", context)
        print("confidence: ", confidence_score.item())
        print("answer: ", answer)
        print("\n\n")

        return {"confidence_score": confidence_score.item(), "answer": answer, "context":context}

    def post_process_output(self, decoded_text):
        # Define a list of punctuation marks to consider
        punctuation_marks = ['.', ',', ';', ':', '!', '?']

        # Use regular expressions to find punctuation tokens with spaces before them
        pattern = r'(\w)\s?(' + r'|'.join(re.escape(p) for p in punctuation_marks) + r')\s'
        processed_text = re.sub(pattern, r'\1\2 ', decoded_text)

        return processed_text

    def get_candidate_docs(self, query_data):
        documents = self.get_documents_from_mongo()

        top_n_documents = self.get_topn_docs(documents_list=documents,
                                             query_data=query_data)

        return top_n_documents

    def get_topn_docs(self, documents_list, query_data):
        query_embedding = np.array(query_data[self.EMBEDDINGS])
        query_tokens = np.array(query_data[self.TOKENS])

        # remove the paddings from the query
        query_embedding = np.array([emb for emb, token in zip(query_embedding, query_tokens) if token != '[PAD]'])

        # List to store cosine similarity scores and corresponding document filenames
        similarity_scores = []

        for doc in documents_list:
            chunk_embeddings = np.array(doc[self.DOCUMENT_EMBEDDING])
            chunk_tokens = np.array(doc[self.DOCUMENT_TOKENS])

            # remove the paddings and unknown tokens from the query
            chunk_embeddings = np.array(
                [emb for emb, token in zip(chunk_embeddings, chunk_tokens) if token not in ['[PAD]', '[UNK]']])

            # Calculate cosine similarity between query_embedding and chunk_embeddings METHOD = 'MEAN_MAX'
            if self.METHOD == 'MEAN_MAX':
                similarity = cosine_similarity(query_embedding, chunk_embeddings)
                similarity = np.mean(np.max(similarity, axis=1))

            elif self.METHOD == 'MEAN_MEAN':
                similarity = cosine_similarity(query_embedding, chunk_embeddings)
                similarity = np.mean(similarity)

            # if self.METHOD == 'COMBINE_MEAN':
            else:
                similarity = cosine_similarity(np.mean(query_embedding, axis=0).reshape(1, -1),
                                               np.mean(chunk_embeddings, axis=0).reshape(1, -1))
                similarity = np.mean(similarity)  # Get the single value out of the array

            # Store similarity score and filename
            similarity_scores.append((similarity, doc))

        # Sort the similarity_scores in descending order based on the similarity score
        if similarity_scores:
            similarity_scores.sort(key=lambda x: x[0], reverse=True)
            return similarity_scores[:self.top_N]
        return similarity_scores

    def get_documents_from_mongo(self):
        # Escape the username and password
        escaped_username = quote_plus(self.username)
        escaped_password = quote_plus(self.password)

        # use MongoDb class to connect to database instance and get the documents
        mongo_db = MongoDb(escaped_username, escaped_password, self.cluster_url,
                           self.database_name, self.collection_name)

        if mongo_db.connect():
            documents = [document for document in mongo_db.iterate_documents()]
            print(f"Total documents: {mongo_db.count_documents()}")
            mongo_db.disconnect()
            return documents
        else:
            return []

    def process_query(self, user_query):
        user_query = user_query.lower()

        # clean query for BERT input
        user_query = clean_text(user_query)

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
        max_query_length = hyperparams["max_query_length"]
        if len(tokenized_query) > max_query_length:
            tokenized_query = tokenized_query[:max_query_length]
        else:
            padding_length = max_query_length - len(tokenized_query)
            tokenized_query = tokenized_query + [self.tokenizer.pad_token] * padding_length

        # Convert the tokenized query to input IDs and attention mask
        input_ids_query = self.tokenizer.convert_tokens_to_ids(tokenized_query)
        attention_mask_query = [1] * len(input_ids_query)

        # Convert to tensors
        input_ids_query = torch.tensor(input_ids_query).unsqueeze(0)  # Add batch dimension
        attention_mask_query = torch.tensor(attention_mask_query).unsqueeze(0)  # Add batch dimension

        # Get the query embeddings for the candidate document search
        query_embeddings = tokens_to_embeddings(tokenized_query, self.model, RANDOM=False)
        query_embeddings_search = tokens_to_embeddings(tokenized_query_for_search, self.model, RANDOM=False)
        query_embeddings_less_sw = tokens_to_embeddings(tokenized_query_for_search_less_sw, self.model, RANDOM=False)

        query_data = {
            "query": user_query,
            "input_ids_query": input_ids_query.tolist(),
            "attention_mask_query": attention_mask_query.tolist(),
            "query_search": user_query_for_search,
            "tokenized_query": tokenized_query,
            "tokenized_query_search": tokenized_query_for_search,
            "tokenized_query_search_less_sw": tokenized_query_for_search_less_sw,
            "query_embedding": query_embeddings.tolist(),  # Just used for the candidate search
            "query_embedding_search": query_embeddings_search.tolist(),  # Just used for the candidate search, cleaned
            "query_embedding_search_less_sw": query_embeddings_less_sw.tolist()
            # Just used for the candidate search, cleaned more
        }
        # return json.dumps(query_data['query'], indent=2)
        return query_data
