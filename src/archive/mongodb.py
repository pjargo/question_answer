import collections
from pymongo import MongoClient
import os, json


class MongoDB(object):

    def __init__(self, mongoclient_dict, database, collection):
        if 'host' not in mongoclient_dict.keys():
            mongoclient_dict['host'] = 'localhost'

        client = MongoClient(**mongoclient_dict)
        db = client[database]
        col = db[collection]

        self.mongoclient_dict = mongoclient_dict
        self.client = client
        self.db = db
        self.col = col
        return

    def load_jsons(self, directory):
        print('creating dataset...')
        parsed_docs = {}
        file_counter = 0
        files = os.listdir(directory)
        files.sort()
        for filename in files:
            #print('File: ' + str(file_counter) + '/' + str(len(os.listdir(directory))-1))
            filepath = os.path.join(directory,filename)
            #print(filepath)
            file_counter += 1
            if filename.endswith('.json'):
                a_file = open(filepath, "r")
                data = json.load(a_file)
                #if data['content_normalized_compound_sentiment'] != 0:
                parsed_docs[data[filename.replace('.json', '')]] = data
                a_file.close()
        print('dataset created.\n') 
        self.jsons = parsed_docs
        return parsed_docs

    def jsons_update_db(self, directory=None):
        if directory is not None:
            self.load_jsons(directory)

        vals = list(self.jsons.values())
        for val in vals:
            self.col.update_one(
                {'sha_256' : val['sha_256']},
                {'$set' : {k: v for k, v in val.items() if k != '_id'}},
                upsert=True
            )

    def update_document(self, doc_dict):
        self.col.update_one(
                {'sha_256' : doc_dict['sha_256']},
                {'$set' : {k: v for k, v in doc_dict.items() if k != '_id'}},
                upsert=True
            )

    def get_document(self, sha256):
        return list(self.col.find({'sha_256' : sha256}))

    def get_all_documents(self):
        cursor = self.col.find({})
        return {document['sha_256'] : document for document in cursor}

    def update_many_documents(self, parsed_docs):
        if isinstance(parsed_docs, dict):
            for val in parsed_docs.values():
                self.update_document(val)
        else:
            for val in parsed_docs:
                self.update_document(val)

    def query_documents(self, query_dict):
        return self.col.find(query_dict)

    def remove_fields(self, fields, query_dict={}):
        if isinstance(fields, dict):
            self.col.update(
                query_dict,
                {'$unset' : fields}
            )
        else:
            self.col.update(
                query_dict,
                {'$unset' : {field:'' for field in fields if field != '_id'}}
            )
        
        