#!/usr/bin/env python
import logging
from pathlib import Path
import time
import json

from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from FLD_user_shared_settings import run_by_engine
from settings import DIENAME_IGNORE_PARAMS

logger = logging.getLogger(__name__)


def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # input_top_dir = Path('./outputs/01.predict.py/2023-08-31.jpn/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/2023-08-31.jpn/')

    # input_top_dir = Path('./outputs/01.predict.py/20230905.LLM_FS/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20230905.LLM_FS/')

    # input_top_dir = Path('./outputs/01.predict.py/20230919.jpn/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/220230919.jpn/')

    # input_top_dir = Path('./outputs/01.predict.py/20231106.refaactor/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231106.refaactor/')

    # input_top_dir = Path('./outputs/01.predict.py/20231107.preliminary/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary/')

    # input_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_samples/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary.many_samples/')

    # input_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_samples.seed--1/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary.many_samples.seed--1/')

    # input_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_samples.seed--2/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary.many_samples.seed--2/')

    # input_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_seeds/')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary.many_seeds/')

    # input_top_dir = Path('./outputs/01.predict.py/20231109.icl_max_proof_by_contradiction_per_label')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231109.icl_max_proof_by_contradiction_per_label')

    # input_top_dir = Path('./outputs/01.predict.py/20231109.3-shot')
    # output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231109.3-shot')

    input_top_dir = Path('./outputs/01.predict.py/20231110.refactor')
    output_top_dir = Path('./outputs/02.evaluate_proofs.py/20231110.refactor')

    skip_if_exists = False
    dry_run = False

    # ------------------------------------ run ------------------------------------
    engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.large')

    for reply_path in input_top_dir.glob('**/replies.jsonl'):
        setting = {
            'input_path': str(reply_path),
        }

        reply_setting = json.load(open(reply_path.parent / 'lab.params.json'))
        setting.update({
            f'{name}': val
            for name, val in reply_setting.items()
        })

        output_dir = build_dir(
            setting,
            top_dir=str(
                Path(output_top_dir)
                / f'dtst_nm={setting.get("dataset_uname", None)}'
            ),
            short=True,
            dirname_ignore_params=DIENAME_IGNORE_PARAMS,
            save_params=True
        )
        
        command = ' '.join([
            'python ./scripts/evaluate_proofs.py',
            str(reply_path),
            str(output_dir),
            # '--similarity-threshold' if similarity_threshold else '',
            # f'--allowed-additional-proof-steps {allowed_additional_proof_steps}',
        ])

        if skip_if_exists and (output_dir / 'metrics_summary.json').exists():
            logger.warning('skip evaluating for the existing results "%s"', str(output_dir))
        else:
            run_by_engine(
                engine,
                command,
                output_dir,
                hours=1,
                dry_run=dry_run
            )

    logger.info('------------- ./02.evaluate_proofs.py finished !! -----------')


if __name__ == '__main__':
    main()
