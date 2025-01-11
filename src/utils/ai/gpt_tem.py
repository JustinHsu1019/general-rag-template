import textwrap

import openai

import utils.config_log as config_log

config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)


def gpt_template(prompt):
    openai.api_key = config.get('OpenAI', 'api_key')

    userprompt = textwrap.dedent(
        f"""
        {prompt}
    """
    )

    response = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': userprompt},
        ],
    )

    return response.choices[0].message['content']


def main():
    """Example: GPT template usage"""
    # import utils.gpt_integration as gpt_call
    # gpt_call.gpt_template()
    print(gpt_template('Question: What are the planets in the solar system? Please return it in json format, {"Return content": "_answer_"}'))


if __name__ == '__main__':
    main()
