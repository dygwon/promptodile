# :crocodile: Promptodile

# Overview
Promptagator demonstrated that Large Language Models (LLMs) with few-shot prompts can be used as task-specific query generators for fine-tuning domain-specialized dense retrieval models. However, the original Promptagator approach relied on proprietary and large-scale LLMs which users may not have access to or may be prohibited from using with sensitive data. In this work, we study the impact of open-source LLMs at accessible scales (<=14B parameters) as an alternative. Our results demonstrate that open-source LLMs as small as 3B parameters can serve as effective Promptagator-style query generators. We hope our work will inform practitioners with reliable alternatives for synthetic data generation and give insights to maximize fine-tuning results for domain-specific applications.

# Citation
```bibtex
@inproceedings{gwon_2025_promptodile,
  title={Study on LLMs for Promptagator-Style Dense Retriever Training},
  author={Gwon, Daniel and Jedidi, Nour and Lin, Jimmy},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  series = {CIKM '25},
  year = {2025},
  pages = {XXX--XXX},
  publisher = {Association for Computing Machinery},
}
```

# Installation
For evaluation of our retrieval methods, we used [Pyserini](https://github.com/castorini/pyserini). Unfortunately, we found some issues when using Pyserini with the other required packages. Thus, we recommend creating separate virtual environments - one for query generation & retriever training and the other for indexing & evaluation.

## Query Generation and Retriever Training
Uses the latest version of python 3.12 and CUDA 12.8 (see [vLLM documentation](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#installation) for details)

```bash
$ # Create your environment.
$ conda create -n promptodile python=3.12 -y
$ conda activate promptodile
$
$ # First, install vllm with torch-backend flag.
$ uv pip install vllm --torch-backend=auto
$
$ # Then, install the rest of your packages.
$ uv pip install -r requirements.txt
```

## Index/Evaluation
To create a custom environment for Pyserini, users should visit [here](https://github.com/castorini/pyserini/blob/master/docs/installation.md#pypi-installation-walkthrough) for detailed instructions.

Note that the optional dependency, `faiss-cpu`, is only intended if you plan to index your corpus on CPU; use `faiss-gpu` to index on GPU. The linux installation instructions are also for CPU; if you have a GPU, please adjust the PyTorch index-url based on your CUDA version. We've found the faiss [installation instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) more useful to install `faiss-cpu` or `faiss-gpu`.

We use Python 3.11 to install Pyserini and CUDA 12.4 to install PyTorch & faiss-gpu. We run evaluation with Pyserini on Linux, and downgrade numpy to 1.26.4.

```bash
$ # See links above for installation instructions for pyserini
$ conda activate pyserini
```

# Usage
Usage can be broken down into three steps:
1. Query Generation
2. Retriever Model Training
3. Index/Evaluation

There is considerable variation in configurations across each of the three steps. In addition to a shared configuration file, each step has its own configuration file.
1. qgen.json
2. train.json
3. eval.json
4. shared.json

Examples have been provided in `./configs/templates` to train [contriever](https://huggingface.co/facebook/contriever) and [e5](https://huggingface.co/facebook/contriever) backbone models. 

## Query Generation
Query generation is designed for offline batched inference using [vLLM](https://docs.vllm.ai/en/stable/). Furthermore, the package is designed to use instruct models, so chat templates should be used for best performance.

```bash
$ conda activate promptodile
$ python -m promptodile.query_generation.generate qgen.json shared.json
```

## Retriever Training
```bash
$ conda activate promptodile
$ accelerate launch --num_processes=GPUS -m promptodile.train train.json shared.json
```

## Index/Evaluation
```bash
$ conda activate pyserini
$
$ # When run as a script, will automatically evaluate and output NDCG@10
$ python -m promptodile.index index.json shared.json
```

# Data
For consistency, we attempt to follow the dataset formatting established by TREC (Text REtrieval Conference) as closely as possible.

## BEIR
Please visit [BEIR](https://huggingface.co/BeIR) for relevant datasets.

You can use utility functions in `promptodile/utils.py` to convert the corpus and queries to TREC format.

## Input Files

### corpus.jsonl
Contain all of the documents found in your corpus. Each json line in the file should contain three of five possible fields:
1. `docid`
2. `url` (not used)
3. `title`
4. `headings` (not used)
5. `body`

[more details](https://trec-rag.github.io/annoucements/2025-rag25-corpus/#document-structure)

### queries.jsonl
Contains the query text that maps to the Topics found in `examples.txt`. At minimum, the query text for each example must be provided in the following format:
1. `id` (mapping to Topic in `examples.txt`)
2. `narrative` (the query/topic's text)

### qrels
This is a text file containing whitespace-delimited rows for documents, topics (or queries), and relevance judgments. No headers are included, but each entry in a row maps to:
1. `Topic`
2. `Iteration`
3. `Document#`
4. `Relevancy`

[more details](https://trec.nist.gov/data/qrels_eng/)

### examples.txt
If provided, represents the few-shot examples to be used in the query generation prompt. Uses the same formatting as the qrels text file.

## Output Files

### syn_queries.jsonl
Generated and uses the same formatting as corpus.jsonl, but adds a `queries` field to each line:
1. `docid`
2. `url` (not used)
3. `title`
4. `headings` (not used)
5. `body`
6. `queries` (generated)

The value for the `queries` field is a list containing each of the synthetic queries generated for the document.

### runs
A text file containing a ranked list of retrieved documents for a set of queries. This is generated after indexing to evaluate the finetuned model. Rows are white-space delimited and the entries in each row correspond to the following headers (not included in the file):
1. `Topic ID`
2. `Q0` (a fixed string)
3. `docid`
4. `Rank`
5. `Score`
6. `Run ID`

[more details](https://trec-rag.github.io/annoucements/2025-track-guidelines/#output-format-ranked-results)

# Disclosure
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
