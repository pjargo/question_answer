from django import template
import re
import html

register = template.Library()


def find_pattern_indices(pattern, text):
    matches = re.finditer(pattern, text)
    indices = [(match.start(), match.end()) for match in matches]
    return indices


@register.filter
def highlight_answer(context, answer_n_confidence):
    highlighted_context = context
    for answer, confidence in answer_n_confidence:
        if context and answer:
            answer = html.unescape(answer)
            # Escape special characters in the answer
            escaped_answer = re.escape(answer.lstrip())
            # Use a regular expression to find and replace the answer in the context
            highlighted_context = re.sub(
                escaped_answer,
                lambda match: f'<span class="highlight">{match.group()}</span>',
                highlighted_context,
                flags=re.IGNORECASE  # Optional: Case-insensitive matching
            )

    return highlighted_context


@register.filter(name='remove_tokens')
def remove_tokens(value):
    # Remove text between the first <s> and </s> tokens
    value = re.sub(r'<s>.*?</s>', '', value, count=1)
    # Remove <s>, </s>, and <pad> tokens
    value = re.sub(r'<s>|</s>|<pad>', '', value)
    return value
