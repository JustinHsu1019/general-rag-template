from utils.ai.gemini_tem import gemini_template
from utils.ai.gpt_tem import gpt_template


def call_llm(content_list, user_input, use_gpt: bool):
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and conversational tone. \

Please reply in english

PASSAGE:
'{content_list}'

[User Question]: '{user_input}'

ANSWER:
"""
    try:
        if use_gpt:
            res = gpt_template(prompt)
        else:
            res = gemini_template(prompt)
    except Exception:
        res = 'Too many user requests! Please wait a few seconds before asking again'

    return res
