# FLD In-Context Learning
This repository includes the code to evaluate large language models on FLD corpora under few-shot in-context learning (Section 5.2 of the paper).  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD.git) about the whole FLD project.

## Installation
The code has been tested on Python 3.11.5
```console
$ pip install -r ./requirements/requirements.txt
$ git clone https://github.com/hitachi-nlp/FLD-task.git && pip install -e ./FLD-task
$ export PYTHONPATH=`pwd -P`:$PYTHONPATH
```

## How to evaluate LLMs

1. Make a dataset that pairs in-context examples and test examples:

    ```console
    $ python ./scripts/make_dataset.py \
        --output-dir ./outputs/dataset \
        --dataset-name hitachi-nlp/FLD.v2 \
        --dataset-config-name default  \
        --n-shot 10 \
        --seed 0
    ```

1. Make predictions by LLMs.

    For OpanAI model, specify model names as "openai.xx":
    ```console
    $ python ./scripts/predict.py \
        ./outputs/dataset/ICL_dataset.jsonl \
        ./outputs/predictions/ \
        --model-name openai.gpt-3.5-turbo-16k \
        --max-examples 5
    ```
    For other models on the huggingface hub, specify model names as "hf.xx":
    ```console
    $ python ./scripts/predict.py \
        ./outputs/dataset/ICL_dataset.jsonl \
        ./outputs/predictions/ \
        --model-name hf.Yukang/Llama-2-13b-longlora-32k-ft \
        --tokenizer-name hf.meta-llama/Llama-2-7b-hf \
        --max-examples 5 \
        --tensor-parallel-size 1
    ```

1. Compute the metircs:
    ```console
    $ python ./scripts/evaluate_proofs.py \
        outputs/predictions/predictions.jsonl \
        outputs/metrics
    ```

1. Analyze predictions:
    ```console
    $ python ./scripts/analyze_results.py \
        ./outputs/metrics/metrics.jsonl \
        ./outputs/analysis
    ```
