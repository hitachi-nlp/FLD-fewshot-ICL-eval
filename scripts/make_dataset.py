#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional, Any
import json
import random
from collections import defaultdict
import textwrap

from logger_setup import setup as setup_logger
from datasets import load_dataset
import click
from FLD_task import (
    load_deduction,
    Deduction,
    SerializedDeduction,
    serialize,
)

# import line_profiling

logger = logging.getLogger(__name__)


# @profile
def load_examples(dataset_name: Optional[str] = None,
                  dataset_config_name: Optional[str] = None,
                  train_file: Optional[str] = None,
                  test_file: Optional[str] = None,
                  reload_deduction=False)\
        -> Tuple[List[Dict], List[Dict]]:
    """ return dict of {split: {label: [example1, example2, ...]}}]}} """

    if dataset_name is not None:
        datasets = load_dataset(dataset_name, dataset_config_name)
    else:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if test_file is not None:
            data_files["test"] = test_file
        datasets = load_dataset('json', data_files=data_files)
    return list(datasets['train']), list(datasets['test'])


def sample_ICL_examples(examples: List[Dict],
                        n_shot: int,
                        reload_deduction=False) -> List[Dict]:
    train_labeled_examples: Dict[str, List[Dict]] = defaultdict(list)
    for dataset_example in examples:
        if reload_deduction:
            deduction = load_deduction(dataset_example)
            label = deduction.world_assump_label
            train_labeled_examples[label].append(deduction.dict())
        else:
            label = dataset_example['world_assump_label']
            train_labeled_examples[label].append(dataset_example)

    ICL_examples: List[Dict] = []
    labels = _shuffle(list(train_labeled_examples.keys()))
    n_shot_per_label = int(n_shot / len(labels))
    if n_shot_per_label > 0:
        for label in labels:
            ICL_examples.extend(random.sample(train_labeled_examples[label], n_shot_per_label))
    for i_remainder in range(n_shot - len(ICL_examples)):
        label = labels[i_remainder % len(labels)]
        ICL_examples.append(random.choice(train_labeled_examples[label]))
    return ICL_examples


def make_ICL_texts(prompt_version: str,
                   ICL_examples: List[Dict],
                   test_example: Dict,
                   reload_deduction=False) -> Tuple[str, str]:
    prompt = '\n\n\n'.join([
        make_intro_text(prompt_version),
        *[make_example_texts(example, reload_deduction=reload_deduction)[0] for example in ICL_examples],
        make_question_text(prompt_version),
        make_example_texts(test_example, hide_output=True, reload_deduction=reload_deduction)[0],

    ])
    gold_proof = make_example_texts(test_example, reload_deduction=reload_deduction)[2]
    return prompt, gold_proof


_INSTRUCTIONS = {

    'v0': {
        'introduction': '==== 1. First, we show some examples of deductive reasoning tasks as follows. In each example, the sentences after the "facts" show a set of facts. Based on these facts, you have to either prove the "hypothesis", disprove it, or declare it as unknown if the facts are insufficient. You have to write the step-by-step thought after the "output".',
        'question': '==== 2. Now, solve the following example. Write the step-by-step thought after the "output" using the same format demonstrated in the above examples.',
    },

    'v1': {
        'introduction': """
                        ******** First, we show some examples of deductive reasoning tasks below. ********
                        An example consists of an input part and an output part, shown after "---- input ----" and "---- output ----," respectively.

                        In the input part, we have a set of facts shown after "$facts$." Based on these facts, we want to verify a hypothesis written after "$hypothesis."

                        The output part shows the step-by-step thought to verify the hypothesis.

                        Each line of the output part shows a fine-grained reasoning step. In each step, the left side of the arrow "->"  shows the set of premises to be used, such as "fact2 & int3" if the set includes the fact numbered as two and the intermediate conclusion numbered as three. The right side of the arrow "->" shows the conclusion that logically follows from the premises. Note that this conclusion should be new, i.e., not match any of the facts or the previously derived conclusions.

                        After these steps, we conclude either the hypothesis can be proved (__PROVED__), disproved (__DISPROVED__), or neither (__UNKNOWN__) because the facts are insufficient.
                        """,
        'question': '******** 2. Now, solve the following example, i.e., write a step-by-step thought to verify the hypothesis after the "output", using exactly the same format demonstrated in the above examples.',
    },

    'v2': {
        'introduction': """
                        ******** First, we show some examples of deductive reasoning tasks below. ********

                        An example consists of an input part and an output part, shown after "---- input ----" and "---- output ----," respectively.

                        In the input part, we have a set of facts shown after "$facts$." Based on these facts, we want to verify a hypothesis written after "$hypothesis."

                        The output part shows the step-by-step thought to verify the hypothesis.

                        Each line of the output part shows a fine-grained reasoning step. In each step, the left side of the arrow "->"  shows the set of premises to be used, such as "fact2 & int3" if the set includes the fact numbered as two and the intermediate conclusion numbered as three. The right side of the arrow "->" shows the conclusion that logically follows from the premises. Note that this conclusion should be new, i.e., not match any of the facts or the previously derived conclusions. For example, the following is not allowed: "fact3 -> int2: this is a sentence" where the content of "fact3" is "this is a sentence".

                        After these steps, we conclude either the hypothesis can be proved (__PROVED__), disproved (__DISPROVED__), or neither (__UNKNOWN__) because the facts are insufficient.
                        """,

        'question': '******** 2. Now, solve the following example, i.e., write a step-by-step thought to verify the hypothesis after the "output", using exactly the same format demonstrated in the above examples.',
    },

}


def make_intro_text(prompt_version: str) -> str:
    return textwrap.dedent(_INSTRUCTIONS[prompt_version]['introduction'])[1:-1]


def make_question_text(prompt_version: str) -> str:
    return textwrap.dedent(_INSTRUCTIONS[prompt_version]['question'])[1:-1]


def make_example_texts(example: Dict, hide_output=False, reload_deduction=False) -> Tuple[str, str, str]:
    if reload_deduction:
        serial = serialize(load_deduction(example),
                           stepwise=False, newlines=True, proof_indicator=False)
        input_text = serial.input
        output_text = serial.gold_proofs[0]
    else:
        input_text = example['prompt_serial']
        output_text = example['proof_serial']

    example_texts = [
        '============ an example ============',
        '---- input ----',
        input_text,
        '---- output ----',
    ]
    if not hide_output:
        example_texts.append(output_text)
    example_text = '\n\n'.join(example_texts)

    return example_text, input_text, output_text


def _shuffle(elems: List[Any]) -> List[Any]:
    return random.sample(elems, len(elems))


@click.command()
@click.option('--output-dir', default=None)
@click.option('--dataset-name', default=None)
@click.option('--dataset-config-name', default='default')
@click.option('--train-file', default=None)
@click.option('--test-file', default=None)
@click.option('--prompt-version',
              type=click.Choice(['v0', 'v1', 'v2']),
              default='v2')
@click.option('--n-shot', type=int, default=10)
@click.option('--seed', type=int, default=0)
@click.option('--icl-max-proof-by-contradiction-per-label', type=int, default=None)
@click.option('--reload-deduction', is_flag=True, default=False)
@click.option('--log-level', default='INFO')
def main(output_dir,
         dataset_name,
         dataset_config_name,
         train_file,
         test_file,
         n_shot,
         prompt_version,
         seed,
         icl_max_proof_by_contradiction_per_label,
         reload_deduction,
         log_level):
    setup_logger(level=log_level, clear_other_handlers=True)
    random.seed(seed)

    train_examples, test_examples = load_examples(dataset_name=dataset_name,
                                                  dataset_config_name=dataset_config_name,
                                                  train_file=train_file,
                                                  test_file=test_file,
                                                  reload_deduction=reload_deduction)

    ICL_examples = sample_ICL_examples(train_examples, n_shot, reload_deduction=reload_deduction)
    ICL_examples = _shuffle(ICL_examples)

    text_examples_no_leak = [example for example in test_examples if example not in ICL_examples]
    text_examples_no_leak = _shuffle(text_examples_no_leak)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'prompts.jsonl', 'w') as f_jsonl, \
            open(output_dir / 'prompts.txt', 'w') as f_txt:
        for test_example in text_examples_no_leak:

            prompt, gold_proof = make_ICL_texts(prompt_version, ICL_examples, test_example,
                                                reload_deduction=reload_deduction)

            instance = {
                'prompt': prompt,
                'gold_proof': gold_proof,
                'ICL_examples': ICL_examples,
                'example': test_example,
                'prompt_num_words': len(prompt.split()),
            }

            f_jsonl.write(json.dumps(instance) + '\n')

            f_txt.write(
                '\n\n\n************************************************ [meta comment] this is an instance ************************************************\n')
            f_txt.write('\n\n\n************************ [meta comment] this is the prompt ************************\n\n')
            f_txt.write(prompt)
            f_txt.write('\n\n\n************************ [meta comment] this is the gold output ************************\n\n')
            f_txt.write(gold_proof)


if __name__ == '__main__':
    main()
