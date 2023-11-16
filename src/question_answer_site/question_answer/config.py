TOKENIZER = "roberta"
INPUT_FOLDER = "space_based_pdfs"
EMBEDDING_MODEL_TYPE = "glove"
EMBEDDING_MODEL_FNAME = "roberta_space_based_pdfs_glove_model.bin"
VECTOR_SIZE = 50
WINDOW = 3
MIN_COUNT = 3
SG = 0
TOKENS_TYPE = "tokens_less_sw"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 0
MAX_QUERY_LENGTH = 20
TOP_N = 10
TOKENS_EMBEDDINGS = "query_search_less_sw"
DOCUMENT_EMBEDDING = "token_embeddings_less_sw"
DOCUMENT_TOKENS = "tokens_less_sw"
METHOD = 'MEAN_MAX'
TRANSFORMER_MODEL_NAME = "deepset/roberta-base-squad2"
CONTEXT_SIZE = 500

username = "new_user_1"
password = "password33566"
cluster_url = "cluster0"
database_name = "question_answer"


special_characters = [
    "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "://", "https",'"', '"...', "/)","www",
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", ".[", ",[", "-,", "][", "com",
    "),", ',"',').'
]

special_characters += list(map(lambda x: "Ġ" + x, special_characters))

# Add numbers to remove
special_characters += list(map(lambda x: str(x), range(100000)))
special_characters += list(map(lambda x: "Ġ" + str(x), range(100000)))