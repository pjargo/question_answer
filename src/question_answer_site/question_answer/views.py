from .question_processing import QuestionAnswer
from django.utils.html import escape
import os
import copy
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import render
from .utils import tokens_to_embeddings
from .embedding_layer import update_embedding_model, get_embedding_model, \
    update_mongo_documents_bulk
from .parse_document import parse_pdf, update_collection
from .config import special_characters, TOKENS_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, \
    username, password, cluster_url, database_name, DOCUMENT_EMBEDDING, TRANSFORMER_MODEL_NAME
from transformers import RobertaTokenizer
from .mongodb import MongoDb
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
import time
from django.http import JsonResponse

# Set the Tokenizer for your specific BERT model variant
tokenizer = RobertaTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME, add_prefix_space=True)

# Escape the username and password
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)


# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the question answer app.")


def handle_uploaded_file(file):
    # Save the uploaded file temporarily
    file_path = os.path.join(settings.MEDIA_ROOT, file.name)
    print(f"Temporarily saving file: {file_path}")
    with default_storage.open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path


def search_view(request):
    query = ""
    highlighted_documents_list = []
    doc_rec_set = {}

    if request.method == 'POST':
        print("POST request: ", request.POST)
        print("FILES request: ", request.FILES)
        # Check if 'query' is present in the POST data (search action)
        action_type = request.POST.get('action_type', '')
        if action_type == 'file':
            add_file_start_time = time.time()
            file = request.FILES.get('file')
            file_path = handle_uploaded_file(file)

            # Get the data in a dictionary
            print("parsing new file into list of dictionaries...")
            parsed_data = parse_pdf(directory=file_path,
                                    embedding_layer_model=get_embedding_model(),
                                    tokenizer=tokenizer,
                                    chunk_size=CHUNK_SIZE,
                                    chunk_overlap=CHUNK_OVERLAP,
                                    additional_stopwords=special_characters)
            # parsed_data <- dict_keys(['chunk_text', 'chunk_text_less_sw', 'tokens', 'tokens_less_sw',
            # 'token_embeddings', 'token_embeddings_less_sw', 'Document', 'Path', 'Text', 'Original_Text', 'sha_256',
            # 'language', 'language_probability', 'counter'])

            if not parsed_data:
                print("Data already exists")

                response_data = {'status': 'existing', 'query_text': query,
                                 'highlighted_documents': highlighted_documents_list,
                                 'documents_if_no_answer': doc_rec_set}  # Include any additional data you need

                return JsonResponse(response_data)

            # Filtered dictionary list -> dataframe
            selected_keys = ["tokens_less_sw", "counter"]  # Select only the desired keys for each dictionary
            selected_dicts = [{key: d[key] for key in selected_keys} for d in parsed_data]
            new_doc_df = pd.DataFrame(selected_dicts)  # Convert to DataFrame

            # Set the embedding layer
            # Get the data from Mongo
            # Get tokens and counter values from all documents in the mongodb

            # Put New data in 'extracted_text' collection
            update_collection(collection="extracted_text", parsed_data=copy.deepcopy(parsed_data))

            # Get existing data in "parsed documents"
            mongodb = MongoDb(escaped_username, escaped_password, cluster_url, database_name,
                              collection_name="parsed_documents")
            if mongodb.connect():
                cursor = mongodb.get_collection().find({}, {TOKENS_TYPE: 1, 'counter': 1, '_id': 0})
                existing_docs_df = pd.DataFrame(list(cursor))  # Put in dataframe

            # Put New data in 'parsed_documents' collection (after retrieving existing data)
            update_collection(collection="parsed_documents", parsed_data=copy.deepcopy(parsed_data))

            # Combine existing data with new data
            combined_df = pd.concat([existing_docs_df, new_doc_df])
            combined_df.to_excel(os.path.join(os.getcwd(), 'tokens.xlsx'))

            # Train embedding model
            update_embedding_model(df=combined_df)

            # Update dataframe with token embeddings from updated embedding layer
            combined_df[DOCUMENT_EMBEDDING] = combined_df[TOKENS_TYPE].apply(tokens_to_embeddings,
                                                                             args=(get_embedding_model(),))
            combined_df.sort_values(by='counter', inplace=True)

            # Apply the function to update MongoDB for each row in the DataFrame
            print("Updating the Mongo Database with new word embeddings...")
            # combined_df.apply(update_mongo_document, args=(mongodb,), axis=1)
            if mongodb.connect():
                print(f"Updating {mongodb.count_documents()} documents...")

                # Split your DataFrame into chunks for bulk updates
                start_time = time.time()

                updated_documents_cnt = 0
                for chunk in np.array_split(combined_df, len(combined_df) // 400):
                    updated_documents_cnt += chunk.shape[0]
                    update_mongo_documents_bulk(chunk, mongodb)
                    print("done chunk")
                print(f"Updated {updated_documents_cnt} documents")
                print(f"Time taken to update mongodb: {time.time() - start_time} seconds")
                mongodb.disconnect()
            print(f"Time taken to add file: {time.time() - add_file_start_time} seconds")

            response_data = {'status': 'complete', 'query_text': query,
                             'highlighted_documents': highlighted_documents_list,
                             'documents_if_no_answer': doc_rec_set}  # Include any additional data you need

            return JsonResponse(response_data)

        else:
            query = request.POST.get('query', '')
            # You can now process the query and send it to the backend for further processing.
            questionAnswer = QuestionAnswer()
            response_data = questionAnswer.answer_question(query)
            results = response_data.get("results", [])
            source_text_dict = response_data.get("source_text_dictionary", {})
            doc_rec_set = response_data.get("no_ans_found", {})

            # Create a dictionary to store highlighted documents
            highlighted_documents = {}

            # Iterate over results and update highlighted_documents
            for result in results:
                document_name = result.get("document", "")
                answer = result.get("answer", "")
                confidence_score = result.get("confidence_score", "")

                # Escape HTML characters in answer and context
                answer = escape(answer)

                if document_name not in highlighted_documents:
                    # If document is not in the dictionary, add it with the first answer
                    highlighted_documents[document_name] = {
                        'original_text': source_text_dict.get(document_name, ""),
                        'highlights': [(answer, confidence_score)],
                        'document': document_name
                    }
                else:
                    # If document is already in the dictionary, append the new answer
                    highlighted_documents[document_name]['highlights'].append((answer, confidence_score))
            # Convert the values of highlighted_documents to a list for easier iteration in the template
            highlighted_documents_list = list(highlighted_documents.values())

    return render(request, 'search/search.html', {
        'query_text': query,
        'highlighted_documents': highlighted_documents_list,
        'documents_if_no_answer': doc_rec_set
    })
