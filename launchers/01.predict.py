#!/usr/bin/env python
import logging
from pathlib import Path
import json

from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from FLD_user_shared_settings import run_by_engine, maybe_option_value
from settings import DIENAME_IGNORE_PARAMS

logger = logging.getLogger(__name__)


def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # input_top_dir = Path('./outputs/10.make_dataset.py/20230701.refactor_distractors/dtst_nm=20230701.finalize.D3/prmpt_typ=in_context_examples.COT.v1/n_sht=10')
    # output_top_dir = Path('./outputs/01.predict.py/20230701.refactor_distractors')

    # input_top_dir = Path('./outputs/10.make_dataset.py/2023-08-31.jpn/')
    # output_top_dir = Path('./outputs/01.predict.py/2023-08-31.jpn/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20230905.LLM_FS/')
    # output_top_dir = Path('./outputs/01.predict.py/20230905.LLM_FS/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20230919.jpn/')
    # output_top_dir = Path('./outputs/01.predict.py/20230919.jpn/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231106.refaactor/')
    # output_top_dir = Path('./outputs/01.predict.py/20231106.refaactor/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231106.preliminary/')
    # output_top_dir = Path('./outputs/01.predict.py/20231106.preliminary/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231107.preliminary/')
    # output_top_dir = Path('./outputs/01.predict.py/20231107.preliminary/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231107.preliminary/')
    # output_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_samples/')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231107.preliminary.seed--1/')
    # output_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_samples.seed--1')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231107.preliminary.seed--2/')
    # output_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_samples.seed--2')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231107.preliminary.many_seeds')
    # output_top_dir = Path('./outputs/01.predict.py/20231107.preliminary.many_seeds')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231109.icl_max_proof_by_contradiction_per_label')
    # output_top_dir = Path('./outputs/01.predict.py/20231109.icl_max_proof_by_contradiction_per_label')

    # input_top_dir = Path('./outputs/10.make_dataset.py/20231109.3-shot')
    # output_top_dir = Path('./outputs/01.predict.py/20231109.3-shot')

    # input_top_dir = Path('./outputs/00.make_dataset.py/20231110.refactor')
    # output_top_dir = Path('./outputs/01.predict.py/20231110.refactor')

    # input_top_dir = Path('./outputs/00.make_dataset.py/20231110.FLD_task_old')
    # output_top_dir = Path('./outputs/01.predict.py/20231110.FLD_task_old')

    # input_top_dir = Path('./outputs/00.make_dataset.py/20231110.FLD_task_old')
    # output_top_dir = Path('./outputs/01.predict.py/20231110.FLD_task_old')

    input_top_dir = Path('./outputs/00.make_dataset.py/20231111')
    output_top_dir = Path('./outputs/01.predict.py/20231111')

    dataset_unames = [
        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        # '20230729.case_study_finalize.D3',
        # '20230729.case_study_finalize.D8',

        'hf.hitachi-nlp/FLD.v2__default',
        # 'hf.hitachi-nlp/FLD.v2__star',

        # ---------------------------------- 20230826.jpn ------------------------------------
        # '20230826.jpn.D3',
        # '20230826.jpn.D8',

        # ---------------------------------- 20230916.jpn ------------------------------------
        # '20230916.jpn.D1_wo_dist',
        # '20230916.jpn.D1',
        # '20230916.jpn.D3',
        # '20230916.jpn.D5',
    ]

    n_shot_list = [
        3,
        # 10,
        # 16,
        # 32,
    ]

    model_names = [
        # ==== openai models (https://platform.openai.com/docs/models) for the openai models.
        # ('openai.gpt-3.5-turbo-16k', None),
        # ('openai.gpt-4', None),                # context=8k, $0.03 / 1K tokens	$0.06 / 1K tokens

        # ==== llama related models

        # -- long context Llama models
        # ('hf.Yukang/Llama-2-7b-longlora-32k-ft', 'hf.meta-llama/Llama-2-7b-hf'),
        # ('hf.Yukang/Llama-2-13b-longlora-32k-ft', 'hf.meta-llama/Llama-2-7b-hf')
        # ('hf.Yukang/LongAlpaca-7B', 'hf.meta-llama/Llama-2-7b-hf')
        ('hf.Yukang/LongAlpaca-13B', 'hf.meta-llama/Llama-2-13b-hf')

        # ==== other models not used
        # ('openai.gpt-3.5-turbo', None),     # XXX context=4k
        # ('openai.gpt-4-32k', None),            # XXX can not access
        # ('openai.gpt-4-32k-0613', None),       # XXX can not access
        # ('openai.gpt-4-1106-preview', None),   # XXX: somehow less than gpt-4

        # -- can not use the following lora models yet, as vLLM does not support them yet. (https://github.com/vllm-project/vllm/issues/1129)
        # ('hf.Yukang/Llama-2-7b-longlora-32k', 'hf.meta-llama/Llama-2-7b-hf'),
        # ('hf.Yukang/Llama-2-13b-longlora-32k', 'hf.meta-llama/Llama-2-7b-hf')
        # ('hf.Yukang/Llama-2-70b-longlora-32k', 'hf.meta-llama/Llama-2-70b-hf'),
        # ('hf.Yukang/LongAlpaca-70B', 'hf.meta-llama/Llama-2-70b-hf')
    ]

    max_samples = 5
    # max_samples = 31
    # max_samples = 61
    # max_samples = 101
    # max_samples = 201
    # max_samples = None

    engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.small')
    # engine = QsubEngine('ABCI', 'rt_G.large')
    # engine = QsubEngine('ABCI', 'rt_AG.small')
    # engine = QsubEngine('ABCI', 'rt_AF')

    # tensor_parallel_size = 4
    tensor_parallel_size = 1

    skip_if_exists = False
    dry_run = False

    # ------------------------------------ run ------------------------------------
    if isinstance(engine, QsubEngine):
        match engine.resource:
            case 'rt_G.small':
                tensor_parallel_size = 1
            case 'rt_G.large':
                tensor_parallel_size = 4
            case 'rt_AG.small':
                tensor_parallel_size = 1
            case 'rt_AF':
                tensor_parallel_size = 8
            case _:
                raise ValueError(f'unknown resource: {engine.resource}')


    prompt_paths = sorted(input_top_dir.glob('**/ICL_dataset.jsonl'))
    for prompt_path in prompt_paths:
        for model_name, tokenizer_name in model_names:
            setting = {
                'input_path': str(prompt_path),
                'model_name': model_name,
                'max_samples': max_samples,
                'tensor_parallel_size': tensor_parallel_size,
                'tokenizer_name': tokenizer_name,
            }

            dataset_setting = json.load(open(prompt_path.parent / 'lab.params.json'))
            if dataset_setting['dataset_uname'] not in dataset_unames:
                continue
            if dataset_setting['n_shot'] not in n_shot_list:
                continue

            setting.update({
                f'{name}': val
                for name, val in dataset_setting.items()
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

            if skip_if_exists and output_path.exists():
                logger.warning('skip evaluating for the existing results "%s"', str(output_path))
            else:
                command = ' '.join([
                    'python ./scripts/predict.py',
                    str(prompt_path),
                    str(output_dir),
                    f'--model-name {setting["model_name"]}',
                    maybe_option_value('--tokenizer-name', setting.get('tokenizer_name', None)),
                    f'--max-examples {setting["max_samples"]}',
                    f'--tensor-parallel-size {setting["tensor_parallel_size"]}',
                    '--dtype half',
                ])

            run_by_engine(
                engine,
                command,
                output_dir,
                hours=1,
                dry_run=dry_run
            )

    logger.info('------------- 01.predict.py finished !! -----------')


if __name__ == '__main__':
    main()
