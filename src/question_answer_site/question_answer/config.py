TOKENIZER = "roberta"
INPUT_FOLDER = "space_based_pdfs"
EMBEDDING_MODEL_TYPE = "glove"
EMBEDDING_MODEL_FNAME = "roberta_space_based_pdfs_glove_model.bin"
VECTOR_SIZE = 50
WINDOW = 3
MIN_COUNT = 3
SG = 0
TOKENS_TYPE = "tokens_less_sw"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 20
MAX_QUERY_LENGTH = 20
TOP_N = 20
TOKENS_EMBEDDINGS = "query_search_less_sw"
DOCUMENT_EMBEDDING = "token_embeddings_less_sw"
DOCUMENT_TOKENS = "tokens_less_sw"
METHOD = "BOTH"  # 'MEAN_MAX', 'COMBINE_MEAN'
TRANSFORMER_MODEL_NAME = "deepset/roberta-base-squad2"
CONTEXT_SIZE = 350

username = "new_user_1"
password = "password33566"
cluster_url = "cluster0"
database_name = "question_answer"

# Aerospace Mongo Credentials
mongo_host = 'e3-dev-services.e3.aero.org'
mongo_port = 31523
mongo_username = 'playground_user'
mongo_password = 'playground123'
mongo_auth_db = 'admin'  # The authentication database
mongo_database_name = 'playground'  # The name of your database

special_characters = [
    "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "://", "https",'"', '"...', "/)","www",
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", ".[", ",[", "-,", "][", "com",
    "),", ',"',').'
]

special_characters += list(map(lambda x: "Ġ" + x, special_characters))

# Add numbers to remove
special_characters += list(map(lambda x: str(x), range(100000)))
special_characters += list(map(lambda x: "Ġ" + str(x), range(100000)))