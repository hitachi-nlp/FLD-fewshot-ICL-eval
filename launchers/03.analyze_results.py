#!/usr/bin/env python
import logging
from pathlib import Path
import time
import json
import shutil 

from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from FLD_user_shared_settings import run_by_engine
from settings import DIENAME_IGNORE_PARAMS

logger = logging.getLogger(__name__)


def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # ----------------- input output paths ---------------------

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20230711.refactor_distractors')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20230711.refactor_distractors')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/2023-08-31.jpn/')
    # output_top_dir = Path('./outputs/03.analyze_results.py/2023-08-31.jpn/')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231106.refaactor/')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231106.refaactor/')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary/')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231107.preliminary/')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary.many_samples/')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231107.preliminary.many_samples/')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231107.preliminary.many_seeds/')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231107.preliminary.many_seeds/')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231110.refactor')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231110.refactor')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231110.FLD_task_old')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231110.FLD_task_old')

    # input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231111')
    # output_top_dir = Path('./outputs/03.analyze_results.py/20231111')

    input_top_dir = Path('./outputs/02.evaluate_proofs.py/20231111.production')
    output_top_dir = Path('./outputs/03.analyze_results.py/20231111.production')

    # ------------------------------------ run ------------------------------------
    dry_run = False
    engine = SubprocessEngine()

    for metrics_path in input_top_dir.glob('**/*jsonl'):
        setting = {
            'input_path': str(metrics_path),
            'reload_deduction': True,
        }

        metrics_setting = json.load(open(metrics_path.parent / 'lab.params.json'))
        setting.update({
            f'{name}': val
            for name, val in metrics_setting.items()
        })

        output_dir = build_dir(
            setting,
            top_dir=str(
                Path(output_top_dir)
                / f'dtst_nm={setting.get("local_dataset_name", None)}'
            ),
            short=True,
            dirname_ignore_params=DIENAME_IGNORE_PARAMS,
            save_params=True,
        )
        
        command = ' '.join([
            'python ./scripts/analyze_results.py',
            str(metrics_path),
            str(output_dir),
            '--reload-deduction' if setting.get('reload_deduction', False) else '',
        ])

        SubprocessEngine().run(f'cp {str(metrics_path.parent / "*")} {str(output_dir)}')
        run_by_engine(
            engine,
            command,
            output_dir,
            hours=1,
            dry_run=dry_run
        )

    logger.info('------------- ./03.analyze_results.py finished !! -----------')

if __name__ == '__main__':
    main()
