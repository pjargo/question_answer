from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from django.utils.html import mark_safe
import re
from .question_processing import QuestionAnswer

# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the question answer app.")


# def highlight_answer(context, answer):
#     if context and answer:
#         # Escape special characters in the answer
#         escaped_answer = re.escape(answer)
#
#         # Use a regular expression to find and replace the answer in the context
#         highlighted_context = re.sub(
#             escaped_answer,
#             lambda match: f'<span class="highlight">{match.group()}</span>',
#             context,
#             flags=re.IGNORECASE  # Optional: Case-insensitive matching
#         )
#
#         return mark_safe(highlighted_context)
#     return ''

def search_view(request):
    query = ""
    results = []

    if request.method == 'POST':
        query = request.POST.get('query', '')
        # You can now process the query and send it to the backend for further processing.
        questionAnswer = QuestionAnswer()
        response_data = questionAnswer.answer_question(query)
        results = response_data.get("results", [])

    return render(request, 'search/search.html', {
        'query_text': query,
        'results': results
    })