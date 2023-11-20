def search_view(request):
    query = ""
    highlighted_documents_list = []
    doc_rec_set = {}

    if request.method == 'POST':
        action_type = request.POST.get('action_type', '')
        if action_type == 'file':
            # Process pdf file

                return render(request, 'search/search.html', {
                                'query_text': query,
                                'highlighted_documents': highlighted_documents_list,
                                'documents_if_no_answer': doc_rec_set,
                                'status': 'complete'
                            })
        else:
            query = request.POST.get('query', '')
            # Process query

    return render(request, 'search/search.html', {
        'query_text': query,
        'highlighted_documents': highlighted_documents_list,
        'documents_if_no_answer': doc_rec_set
    })

def get_answer_and_confidence(chunks):
    """
    :param chunks: list of tokenized chunks of text
    """
    with torch.no_grad():
        for chunk in chunks:
            tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2", add_prefix_space=True)
            model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

            inputs = tokenizer.encode_plus(query_tokens, chunk, max_length=512, return_tensors="pt",
                                           padding="max_length", truncation=True)
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            start_logits = output.start_logits
            end_logits = output.end_logits

            combined_scores = start_logits.unsqueeze(-1) + end_logits.unsqueeze(1)

            # Find the indices with the highest combined score
            max_combined_score_idx = torch.argmax(combined_scores)

            # Convert the index to start and end indices
            start_idx = torch.div(max_combined_score_idx, combined_scores.size(1), rounding_mode='trunc')
            end_idx = max_combined_score_idx - start_idx * combined_scores.size(1)

            answer_tokens = input["input_ids"][0][start_idx: end_idx + 1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

            if answer == "":
                answer = "Sorry, I don't have information on that topic."

            start_probs = torch.softmax(start_logits, dim=1)
            end_probs = torch.softmax(end_logits, dim=1)
            confidence_score = start_probs[0, start_idx] * end_probs[0, end_idx]

            context = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input["input_ids"][0]))

            return {"confidence_score": confidence_score.item(), "answer": answer, "context": context}