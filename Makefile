all: matmul_dispatch transmission_scheduler example_dataset 

matmul_dispatch: 
	cd awsm_transformer/ops/matmul_helper && pip install -e .

# attention_helper: 
# 	cd awsm_transformer/ops/attention_helper && pip install -v -e .

transmission_scheduler: 
	cd transmission_scheduler_pkg && pip install -e .

example_dataset: 
	pip install tiktoken datasets
