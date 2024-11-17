python sql_rl_gen/generation/evaluate_model.py \
       --model_name_or_path "juierror/flan-t5-text2sql-with-schema-v2" \
       --dataset "example_text2sql_$1_dev" \
       --template llama3 \
       --trained_agent_path "./output/model_$1_train/best" \
       --outdir "./output/model_base_train" \
       --dataset_name $1