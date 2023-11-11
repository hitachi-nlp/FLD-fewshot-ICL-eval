#!/usr/bin/env python
import os
import re
import logging
from pathlib import Path
from typing import List, Union, Dict, Optional
import json
import time

from tqdm import tqdm
from logger_setup import setup as setup_logger
import click

from dotenv import load_dotenv
load_dotenv()  # XXX!! MUST BE PLACED BEFORE langchain and openai
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.llms import VLLM
import openai


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.argument('output_dir')
@click.option('--model-name', default='openai.gpt-4',
              help='choose from https://platform.openai.com/docs/models/model-endpoint-compatibility')
@click.option('--tokenizer-name', default=None)
@click.option('--api-key', default=None)
@click.option('--max-examples', type=int, default=None)
@click.option('--tensor-parallel-size', type=int, default=None)
@click.option('--dtype', default='auto')
@click.option('--log-level', default='INFO')
def main(input_path,
         model_name,
         tokenizer_name,
         output_dir,
         api_key,
         max_examples,
         tensor_parallel_size,
         dtype,
         log_level):
    setup_logger(level=log_level, clear_other_handlers=True)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_service, _model_name = model_name.split('.', 1)

    if model_service == 'openai':
        api_key = api_key or os.environ.get('OPENAI_API_KEY', None)
        if api_key is None:
            raise ValueError('API key is not specified.')

        chat_model = ChatOpenAI(model_name=_model_name,
                                openai_api_key=api_key)

        def predict(prompt: str) -> Optional[str]:
            last_exception = None
            for i in range(0, 5):
                try:
                    return chat_model([HumanMessage(content=prompt)]).content
                except openai.error.RateLimitError as e:
                    last_exception = e
                    logger.info('getting a prediction failed due to the following rate limit exception. will sleep  1min and retry:\n%s', str(e))
                    time.sleep(60)
            raise Exception('Could not get any prediction. The last exception is the following:\n' + str(last_exception))

    elif model_service == 'hf':
        # hf = HuggingFacePipeline.from_model_id(
        #     model_id=model_name.lstrip('hf.'),
        #     task='text-generation',
        #     model_kwargs={'trust_remote_code': True, 'use_auth_token': True},
        # )

        # def get_reply(prompt: str) -> Optional[str]:
        #     return hf.generate([prompt]).generations[0][0].text

        llm = VLLM(
            model=model_name.lstrip('hf.'),
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=2000,
            # top_k=10,
            # top_p=0.95,
            # temperature=0.8,
            vllm_kwargs={
                'tokenizer': re.sub('^hf\.', '', tokenizer_name) if tokenizer_name is not None else None,
            },
            # dtype='auto',  # bf16 if possible else fp16
            dtype=dtype,
        )

        def predict(prompt: str) -> Optional[str]:
            # llm("What is the future of AI?")
            # return llm(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
            return llm(prompt)

    else:
        raise ValueError(f'Unknown model service {model_service}')

    output_path = output_dir / 'predictions.jsonl'
    with open(output_path, 'w') as f_out:
        for i_example, line in tqdm(enumerate(open(input_path))):
            if i_example >= max_examples:
                break
            example = json.loads(line.strip('\n'))
            prompt = example['prompt']

            logger.info('-- running the LLM on a example ... --')
            logger.info('prompt # words: %d', len(prompt.split()))

            prediction = predict(prompt)

            if prediction is not None:
                logger.info('prediction # words: %d', len(prediction.split()))

            example['prediction'] = prediction
            f_out.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    main()


