from django.http import HttpResponse
from django.shortcuts import render
from .question_processing import QuestionAnswer
from django.utils.html import escape

# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the question answer app.")


def search_view(request):
    query = ""
    results = []
    source_text_dict = {}
    highlighted_documents_list = []
    doc_rec_set = {}

    if request.method == 'POST':
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
        'documents_if_no_answer':doc_rec_set
    })

