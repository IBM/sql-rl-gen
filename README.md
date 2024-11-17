# SQL-RL-GEN
Large Language Models (LLMs) have revolutionized text and code generation tasks, but the text-to-SQL (text2SQL) problem still remains challenging. Current state-of-the-art models require extensive preprocessing steps to achieve accurate SQL query generation, which can be data-hungry and time-consuming.  We introduce a Reinforcement Learning-based approach that improves text2SQL generation while minimizing resources and maximizing flexibility. The algorithm is based on a Reinforcement Learning approach with a reward function generated by a LLM to guide the agent's training process in solving the specific text2SQL generation task with high effectiveness. The experiments demonstrate an improvement in accuracy of 7% on state-of-the-art SQL generation method with limited training dataset composed of only 1000 samples and small models with 248M parameters.
## Installation & Run
Everything is done in project directory after clone. So, if you run ``ls`` command you will see:
```
$ ls
configs  data_preprocess  LICENSE  README.md  requirements.txt  scripts  setup.py  sql_rl_gen
```
### 1. Conda set-up:
1. 
```shell
conda create -n sql-rl-gen python=3.10
```
2. 
```
The following NEW packages will be INSTALLED:
Proceed ([y]/n)? y
```
3. 
```shell
conda activate sql-rl-gen
```
4. 
```shell
pip install -e .
```
``sql-rl-gen.egg-info`` compiled directory should appear
### 2. Data:
1. Download data from: https://ibm.box.com/v/sql-rl-gen-data
2. Unzip in ``./data_preprocess/data``. 
The files should be organised this way:
```
sql-rl-gen
|     configs
|     data_preprocess
|     |   data
|     |    |    spider
|     |    |    |    database
|     |    |    |    |    ...
|     |    |    wikisql
|     |    |    |    database
|     |    |    |    |    ...
|     |    |    |    dev_tok.jsonl
|     |    |    |    ...
|     |    |    dataset_info.json
|     |    data_utils.py
|     |    ...
|     scripts
|     sql_rl_gen
|     ...
```
3. Construct the data for training and testing. Run:
```shell
chmod +rwx ./scripts/generate_data.sh
./scripts/generate_data.sh spider
```
To generate data on spider dataset.

Run:
```shell
chmod +rwx ./scripts/generate_data.sh
./scripts/generate_data.sh wikisql
```
To generate data on wikisql dataset. This might take some time. As the result, ``example_text2sql_{dataset_name}_{train/test/dev}.json`` files are created inside the ``./data_preprocess/data directory``.
### 3. Run Eureka algorithm to get the best reward function
Put the best generated reward function in ``sql_rl_gen/generation/envs/sql_generation_environment.py`` and run:
```shell
chmod +rwx ./scripts/eureka_sql.sh
./scripts/eureka_sql.sh 4 10 "llama3"
```
### 4. Run train
```shell
chmod +rwx ./scripts/run_train.sh spider
./scripts/run_train.sh
```
After the training is finished, ``./output/model_spider_train`` directory is created, with other folders inside.
### 5. Run evaluation
It depends on what is created in a previous point. You might need to change the ``--trained_agent_path`` parameter to the directory with ``model.pt`` and ``optimizer.pt``
```shell
chmod +rwx ./scripts/evaluate_model.sh
./scripts/evaluate_model.sh spider
```
After the evaluation is finished, it will put files: ``feedback_metrics.csv`` and ``statistics_metrics.csv`` inside the ``./output./{model_name}``
## Technical reference

**Memory**: 32 GB

**Processor**: Intel Core i7-10750H CPU @ 2.60GHz * 12

**Operating System**: Red Hat Enterprise Linux 8.10 64-bit

**Graphics**: NVIDIA Quadro T1000/PCle/SSE2

**CUDA Version**: 12.5
