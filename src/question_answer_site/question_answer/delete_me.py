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