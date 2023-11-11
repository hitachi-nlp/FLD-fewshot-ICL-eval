#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Union, Dict
import json

from logger_setup import setup as setup_logger
import click
from FLD_task import prettify_proof_text, prettify_context_text,  load_deduction


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.argument('output_dir')
@click.option('--reload-deduction', is_flag=True)
@click.option('--log-level', default='INFO')
def main(input_path, output_dir, reload_deduction, log_level):
    setup_logger(level=log_level, clear_other_handlers=True)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = [
        json.loads(line.strip('\n'))
        for line in open(input_path)
    ]
    metrics_types = list(examples[0]['metrics'].keys())

    for metric_type in metrics_types:
        for metric in ['proof_accuracy.zero_one', 'answer_accuracy']:
            errors_path = output_dir / f'metric_type--{metric_type}__metric--{metric}__wrong.txt'
            corrects_path = output_dir / f'metric_type--{metric_type}__metric--{metric}__correct.txt'
            with open(errors_path, 'w') as f_err, open(corrects_path, 'w') as f_corr:

                for i_example, example in enumerate(examples):
                    if metric_type not in example['metrics']:
                        f_err.write('\n\n\n\n\n')
                        f_err.write(f'****************************************** example-{i_example} ******************************************')
                        f_err.write('metrics not found in this example, might be failed to calculate the metrics due to some errors.')
                        continue

                    accuracy = example['metrics'][metric_type][metric]
                    if accuracy > 0.0:
                        f_out = f_corr
                    else:
                        f_out = f_err

                    if reload_deduction:
                        deduction_example = load_deduction(example['example'])
                        facts = deduction_example.context
                        hypothesis = deduction_example.hypothesis
                    else:
                        facts = example['example'].get('facts', example['example'].get('context', None))
                        hypothesis = example['example']['hypothesis']
                    proof_gold = example['gold_proof']
                    proof_pred = example['prediction']

                    f_out.write('\n\n\n\n\n')
                    f_out.write(f'****************************************** example-{i_example} ******************************************')

                    f_out.write('\n\n===================== facts =====================\n')
                    f_out.write(prettify_context_text(facts))

                    f_out.write('\n\n===================== hypothesis =====================\n')
                    f_out.write(hypothesis)

                    f_out.write('\n\n===================== proof_gold =====================\n')
                    f_out.write(prettify_proof_text(proof_gold))

                    f_out.write('\n\n===================== proof_pred =====================\n')
                    f_out.write(prettify_proof_text(proof_pred))


if __name__ == '__main__':
    main()
