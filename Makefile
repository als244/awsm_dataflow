all: create_dirs matmul_dispatch transmission_scheduler 

create_dirs:
	mkdir -p init_models fineweb_ckpts

matmul_dispatch: 
	cd awsm_transformer/ops/matmul_helper && pip install -e .

transmission_scheduler: 
	cd transmission_scheduler_pkg && pip install -e .
