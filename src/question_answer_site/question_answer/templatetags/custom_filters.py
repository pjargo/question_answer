from django import template
import re

register = template.Library()


@register.filter
def highlight_answer(context, answer):
    if context and answer:
        # Escape special characters in the answer
        escaped_answer = re.escape(answer)

        # Use a regular expression to find and replace the answer in the context
        highlighted_context = re.sub(
            escaped_answer,
            lambda match: f'<span class="highlight">{match.group()}</span>',
            context,
            flags=re.IGNORECASE  # Optional: Case-insensitive matching
        )

        return highlighted_context

