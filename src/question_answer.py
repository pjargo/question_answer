from transformers import BertTokenizer, BertForQuestionAnswering, RobertaTokenizer, RobertaForQuestionAnswering
import torch
import json
import os
import re
import sys

def post_process_output(decoded_text):
    # Define a list of punctuation marks to consider
    punctuation_marks = ['.', ',', ';', ':', '!', '?']

    # Use regular expressions to find punctuation tokens with spaces before them
    pattern = r'(\w)\s?(' + r'|'.join(re.escape(p) for p in punctuation_marks) + r')\s'
    processed_text = re.sub(pattern, r'\1\2 ', decoded_text)

    return processed_text


def print_answer(input, output, confidence_threshold=0):
	start_logits = output.start_logits
	end_logits = output.end_logits
	# print(output)
	# print(type(start_logits))
	# print(start_logits)

	# start_idx = torch.argmax(start_logits)
	# end_idx = torch.argmax(end_logits)

	combined_scores = start_logits.unsqueeze(-1) + end_logits.unsqueeze(1)

	# Find the indices with the highest combined score
	max_combined_score_idx = torch.argmax(combined_scores)
	print(combined_scores.size())

	# Convert the index to start and end indices
	start_idx = torch.div(max_combined_score_idx, combined_scores.size(1), rounding_mode='trunc')
	end_idx = max_combined_score_idx - start_idx * combined_scores.size(1)

	print(f"start and end logits: ", start_idx, end_idx)

	answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
	answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
	answer = post_process_output(answer)

	# confidence_score = combined_scores[0, start_idx, end_idx]

	start_probs = torch.softmax(start_logits, dim=1)
	end_probs = torch.softmax(end_logits, dim=1)
	confidence_score = start_probs[0, start_idx] * end_probs[0, end_idx]

	print("Question: ", query_text)
	# Check if the confidence score is below the threshold
	if confidence_score < confidence_threshold:
	    print("Sorry, I don't have information on that topic.")
	else:
	    print("Answer:", answer)
	    print()
	    print("Confidence:", confidence_score)

if __name__ == '__main__':
	MERGE = False
	candidate_responses = dict()

	print(f"running for model:{sys.argv[1]}")
	if sys.argv[1].lower() in ['bert', 'bert-base-uncased', 'bert_base']:
		# Load the pre-trained BERT model and tokenizer
		model_name = "bert-base-uncased"
		tokenizer = BertTokenizer.from_pretrained(model_name)
		model = BertForQuestionAnswering.from_pretrained(model_name)
	else:
		# Load the pre-trained RoBERTa model and tokenizer
		model_name = "deepset/roberta-base-squad2"
		tokenizer = RobertaTokenizer.from_pretrained(model_name)
		model = RobertaForQuestionAnswering.from_pretrained(model_name)

	# Specify the directory path containing JSON files
	candidate_docs_dir = os.path.join('..', 'candidate_docs')
	candidate_docs_dir_list = os.listdir(candidate_docs_dir)
	candidate_docs_dir_list.sort(key=lambda x: os.path.getmtime(os.path.join(candidate_docs_dir, x)), reverse=True)
	candidate_docs_subdir = os.path.join(candidate_docs_dir, candidate_docs_dir_list[0])
	candidate_docs_subdir_list = [file for file in os.listdir(candidate_docs_subdir) if file.endswith('json')]
	candidate_docs_subdir_list.sort(key=lambda x: int(x.split(".")[0]))	# Sort the files such that content maintains order
	print("candidate docs dir: ", candidate_docs_subdir_list)

	# Specify the query directory and get the latest query
	query_dir = os.path.join('..', 'query')
	query_dir_list = os.listdir(query_dir)
	query_dir_list.sort(key= lambda x: os.path.getmtime(os.path.join(query_dir, x)), reverse=True)
	query_dir_fname = os.path.join(query_dir, query_dir_list[0])
	print("query_dir: ", query_dir_fname)

	# Load tokenized query and candidate documents from JSON files
	with open(query_dir_fname, "r") as query_file:
	    query_data = json.load(query_file)
	    query_tokens = query_data["tokenized_query"]
	    query_text = query_data["query"]

	# Concatenate tokens from all candidate chunks
	candidate_docs_tokens_concatenated = []
	prev_doc = None   
	for candidate in candidate_docs_subdir_list:
	    with open(os.path.join(candidate_docs_subdir,candidate), "r") as docs_file:
	        candidate_docs_data = json.load(docs_file)
	        candidate_docs_tokens = candidate_docs_data["tokens"]

	        # Add a seperation token between documents unless they are consecutive numbers, indicating they are not seperate chunks
	        if prev_doc and int(candidate.split(".")[0]) != int(prev_doc.split(".")[0]) + 1:
	        	print(tokenizer.decode(tokenizer.sep_token_id))
	        	candidate_docs_tokens_concatenated.extend([tokenizer.sep_token_id]) 
	        candidate_docs_tokens_concatenated.extend(candidate_docs_tokens)
	        prev_doc = candidate

	chunk_size = 350  # Choose an appropriate chunk size
	chunks = [candidate_docs_tokens_concatenated[i:i+chunk_size] for i in range(0, len(candidate_docs_tokens_concatenated), chunk_size)]

	# Initialize lists to store start and end logits for each chunk
	start_logits_list = []
	end_logits_list = []

	# Process each chunk separately and store logits
	with torch.no_grad():
	    for i, chunk in enumerate(chunks):
	        inputs = tokenizer.encode_plus(query_tokens, chunk, max_length=512, return_tensors="pt", padding="max_length", truncation=True)

	        # Roberta Model does not have token_type_ids as an input argument
	        if sys.argv[1].lower() in ['bert', 'bert-base-uncased', 'bert_base']:
	        	outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"])
	        else:
	        	outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
	        
	        start_logits_list.append(outputs.start_logits)
	        end_logits_list.append(outputs.end_logits)

	        # store the outputs for each chunk
        	candidate_responses[i] = { "inputs": inputs,
        							   "ouptut": outputs,
        							   "start_logits":outputs.start_logits,
        							   "end_logits":outputs.end_logits,
        							   "confidence_score": torch.max(torch.softmax(outputs.start_logits, dim=1)) 
        							   }
	        print("\n\n\n\n\n\n\n")
	        print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])))
	        print()
	        print_answer(inputs, outputs, 0)

	if MERGE:
		# Merging responses from all chunks
		print("start: ", list(map(torch.argmax, start_logits_list)))
		print("end: ", list(map(torch.argmax, end_logits_list)))

		# Concatenate start and end logits from all chunks
		all_start_logits = torch.cat(start_logits_list, dim=1)
		all_end_logits = torch.cat(end_logits_list, dim=1)

		# Get the start and end positions of the answer span
		start_logits, end_logits = all_start_logits, all_end_logits

		# TODO: WHAT ARE THE INPUTS FOR CONCAT??
		# inputs = ???

	else:
		# Get response with thee highest confidence
		print("Getting the best response...")

		confidence_score = float("-inf")
		best_response = None
		for i, output_dict in candidate_responses.items():
			if output_dict["confidence_score"] > confidence_score:
				confidence_score = output_dict["confidence_score"]
				best_response = i

		# print("Best Response: ", best_response)

		inputs = candidate_responses[best_response]["inputs"]
		outputs = candidate_responses[best_response]["ouptut"], candidate_responses[best_response]["end_logits"]

	# print_answer(inputs, outputs, 0)
	