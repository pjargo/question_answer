from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient
from pymongo import errors


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

    def get_documents(self, query, inclusion=None):
        collection = self.get_collection()
        if collection is not None:
            try:
                if inclusion:
                    result = collection.find(query, inclusion)
                else:
                    result = collection.find(query)
                return result
            except Exception as e:
                print(f"Error inserting document: {e}")
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

    def update_document(self, query, update):
        collection = self.get_collection()
        if collection is not None:
            try:
                collection.update_one(query, update)
                return
            except Exception as e:
                print(f"Error updating document: {e}")
        return None

    def bulk_update_documents(self, operations):
        collection = self.get_collection()
        if collection is not None:
            try:
                result = collection.bulk_write(operations)
                return result
            except errors.BulkWriteError as bwe:
                print(f"Bulk write error: {bwe.details}")
            except Exception as e:
                print(f"Error updating documents: {e}")
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

    def get_cursor_all(self):
        collection = self.get_collection()
        if collection is not None:
            return collection.find({})

    def iterate_documents(self):
        collection = self.get_collection()
        if collection is not None:
            cursor = collection.find({})
            for document in cursor:
                yield document
            # return cursor