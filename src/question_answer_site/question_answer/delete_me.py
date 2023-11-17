def search_view(request):
    query = ""
    highlighted_documents_list = []
    doc_rec_set = {}

    if request.method == 'POST':
        print(request.POST)
        print(request.FILES)
        # Check if 'query' is present in the POST data (search action)
        action_type = request.POST.get('action_type', '')
        if action_type == 'file':
            file = request.FILES.get('file')
            print(file)
            print(file.name)
            file_path = handle_uploaded_file(file)

            # Get the data in a dictionary
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
                return HttpResponse('Documents already added')
            print(parsed_data[0]['counter'])
            print(parsed_data[0]['tokens_less_sw'])

            # Select only the desired keys for each dictionary
            selected_keys = ["tokens_less_sw", "counter"]
            selected_dicts = [{key: d[key] for key in selected_keys} for d in parsed_data]

            # Convert the list of selected dictionaries to a DataFrame
            new_doc_df = pd.DataFrame(selected_dicts)
            print("New data:")
            print(new_doc_df.columns)
            print(type(new_doc_df["tokens_less_sw"][0]))
            print()

            # Set the embedding layer
            # Get the data from Mongo
            # Get tokens and counter values from all documents in the mongodb

            # Create a MongoClient and connect to the server
            mongodb = MongoDb(escaped_username, escaped_password, cluster_url, database_name,
                              collection_name="parsed_documents")
            if mongodb.connect():
                # Get all existing data from 'parsed_documents' collection
                cursor = mongodb.get_collection().find({}, {TOKENS_TYPE: 1, 'counter': 1, '_id': 0})

                # Put in dataframe
                existing_docs_df = pd.DataFrame(list(cursor))
                print("Mongo Data:")
                print(existing_docs_df.columns)
                print(type(existing_docs_df["tokens_less_sw"][0]))
                print()
                # Add new data to 'parsed_documents' collection
                doc_cnt = mongodb.count_documents()
                print(f"{doc_cnt} documents in 'parsed_documents' before adding")

                for data_obj in copy.deepcopy(parsed_data):
                    # 'extracted_text'
                    data_obj.pop('Original_Text')
                    data_obj.pop('Text')

                    # -
                    data_obj.pop('language')
                    data_obj.pop('language_probability')
                    data_obj.pop('Path')
                    data_obj.pop('token_embeddings')
                    data_obj.pop('chunk_text')
                    data_obj.pop('chunk_text_less_sw')

                    mongodb.insert_document(data_obj)

                print("Data inserted successfully!")

                doc_cnt = mongodb.count_documents()
                print(f"{doc_cnt} documents in 'parsed_documents' after adding")

                # Close the MongoDB client when done
                mongodb.disconnect()

            # Put New data in 'extracted_text' collection
            # Create a MongoClient and connect to the server
            mongodb = MongoDb(escaped_username, escaped_password, cluster_url, database_name, collection_name="extracted_text")
            if mongodb.connect():

                doc_cnt = mongodb.count_documents()
                print(f"{doc_cnt} documents in 'extracted_text' before adding")

                document_tracker = set()
                for data_obj in copy.deepcopy(parsed_data):
                    # -
                    data_obj.pop('language')
                    data_obj.pop('language_probability')
                    data_obj.pop('Path')
                    data_obj.pop('token_embeddings')
                    data_obj.pop('chunk_text')
                    data_obj.pop('chunk_text_less_sw')

                    # 'parsed_documents'
                    data_obj.pop('counter')
                    data_obj.pop('token_embeddings_less_sw')
                    data_obj.pop('tokens_less_sw')
                    data_obj.pop('tokens')

                    # Insert the JSON data as a document into the collection
                    if data_obj['Document'] not in document_tracker:
                        document_tracker.add(data_obj['Document'])
                        print(data_obj['Document'])
                        mongodb.insert_document(data_obj)
                print("Data inserted successfully!")
                doc_cnt = mongodb.count_documents()
                print(f"{doc_cnt} documents in 'extracted_text' after adding")

            # Close the MongoDB client when done
            mongodb.disconnect()

            combined_df = pd.concat([existing_docs_df, new_doc_df])
            print("combined data")
            print(combined_df.columns)
            # print(combined_df)
            # print(os.path.join(os.getcwd(), 'tokens.xlsx'))
            combined_df.to_excel(os.path.join(os.getcwd(), 'tokens.xlsx'))
            # Train Word2Vec model
            if EMBEDDING_MODEL_TYPE == 'Word2Vec':
                kwargs = {
                    'sentences': combined_df[TOKENS_TYPE].to_list(),
                    'vector_size': VECTOR_SIZE,
                    'window': WINDOW,
                    'min_count': MIN_COUNT,
                    'sg': SG
                }

                # Train the Word2Vec model
                model = Word2Vec(**kwargs)

                # Save the model
                model.save(os.path.join("..", "models", "word_embeddings", EMBEDDING_MODEL_FNAME))

            elif EMBEDDING_MODEL_TYPE == 'glove':
                # Specify the file path for the output text file
                output_file = os.path.join(os.getcwd(), "question_answer", "embedding_models", "glove",
                                           'training_data.txt')

                # Write the "tokens" column to a text file with each row on a separate line
                combined_df[TOKENS_TYPE].apply(lambda x: ' '.join(x)).to_csv(output_file, header=False, index=False,
                                                                             sep='\n',
                                                                             quoting=csv.QUOTE_NONE)

                os.environ["VECTOR_SIZE"] = str(VECTOR_SIZE)
                os.environ["WINDOW_SIZE"] = str(WINDOW)
                os.environ["VOCAB_MIN_COUNT"] = str(MIN_COUNT)
                # sys.path.append(os.path.join("..", "models", "word_embeddings", "glove"))

                # Train the model
                demo_path = os.path.join(os.getcwd(), "question_answer", "embedding_models", "glove")
                os.chdir(demo_path)
                script_path = os.path.join(demo_path, "demo.sh")
                try:
                    # Run the demo.sh script
                    subprocess.run([script_path], check=True, shell=True)
                    # For example: subprocess.run([script_path, 'arg1', 'arg2'], check=True, shell=True)
                except subprocess.CalledProcessError as e:
                    # Handle errors if the subprocess returns a non-zero exit code
                    print(f"Error running script: {e}")
                if os.getcwd().endswith('glove'):
                    views_path = os.path.join("..", "..", "..")
                    os.chdir(os.path.join(views_path))

                # Path to your GloVe vectors file
                vectors_file = os.path.join(os.getcwd(), "question_answer", "embedding_models", "glove", "vectors.txt")

                # Load the custom spaCy model with GloVe vectors
                custom_nlp = load_custom_vectors(vectors_file)

                # Save the custom spaCy model to a directory
                custom_nlp.to_disk(os.path.join(os.getcwd(), "question_answer", "embedding_models", EMBEDDING_MODEL_FNAME.split(".bin")[0]))

                print("updated the embedding layer")

            # Update dataframe with token embeddings
            combined_df[DOCUMENT_EMBEDDING] = combined_df[TOKENS_TYPE].apply(tokens_to_embeddings, args=(get_embedding_model(),))
            combined_df.sort_values(by='counter', inplace=True)
            print(combined_df.shape)
            # Apply the function to update MongoDB for each row in the DataFrame
            # combined_df.apply(update_mongo_document, args=(mongodb,), axis=1)

            # Split your DataFrame into chunks for bulk updates
            start_time = time.time()

            chunk_size = 200  # Adjust this based on your needs
            for chunk in np.array_split(combined_df, len(combined_df) // chunk_size):
                print("chunk shape: ", chunk.shape)
                update_mongo_documents_bulk(chunk, mongodb)
                print("done chunk")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken to update mongodb: {elapsed_time} seconds")

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