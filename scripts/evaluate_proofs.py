#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Union, Dict
import json
from collections import defaultdict
import statistics
from pprint import pprint

from logger_setup import setup as setup_logger
import click
from FLD_task import build_metrics


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.argument('output_dir')
@click.option('--log-level', default='INFO')
def main(input_path,
         output_dir,
         log_level):
    setup_logger(level=log_level, clear_other_handlers=True)
    logging.getLogger('absl').setLevel('WARNING')

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    calc_metric_funcs = {
        type_: build_metrics(type_)
        for type_ in ['strict', 'allow_extra_steps']
    }

    metrics_path = output_dir / 'metrics.jsonl'
    all_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda : defaultdict(list))
    with open(metrics_path, 'w') as f_out:
        for i_example, line in enumerate(open(input_path)):
            example = json.loads(line.strip('\n'))

            gold = example['gold_proof']
            pred = example['prediction']
            if pred is None:
                logger.warning('The example will be skipped because the prediction is None')
                continue
            facts = example['example'].get('facts', example['example'].get('context', None))

            example['metrics'] = {}
            for metric_type, calc_metrics in calc_metric_funcs.items():
                try:
                    metrics = calc_metrics(
                        [gold],
                        pred,
                        context=facts,
                    )
                except Exception as e:
                    logger.warning('The example will be skipped because the following exception was raised:\n%s', str(e))
                    continue

                example['metrics'][metric_type] = metrics

                for name, val in metrics.items():
                    all_metrics[metric_type][name].append(val)

            f_out.write(json.dumps(example) + '\n')

    metrics_summary = defaultdict(lambda: {})
    for metric_type, _all_metrics in all_metrics.items():
        for name, vals in _all_metrics.items():
            metrics_summary[metric_type][name] = statistics.mean(vals)

    json.dump(metrics_summary,
              open(str(output_dir / 'metrics_summary.json'), 'w'),
              ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    pprint(dict(metrics_summary))


if __name__ == '__main__':
    main()
