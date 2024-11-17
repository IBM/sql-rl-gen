python sql_rl_gen/generation/sql_generation.py \
       --model_name_or_path "juierror/flan-t5-text2sql-with-schema-v2" \
       --dataset "example_text2sql_$1_train" \
       --template llama3 \
       --outdir "./output/model_$1_train" \
       --steps_n 1000 \
       --dataset_name $1