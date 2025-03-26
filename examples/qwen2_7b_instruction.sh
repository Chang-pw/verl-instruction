set -x

expoert VLLM_ATTENTION_BACKEND=XFORMERS

# ray stop
# sleep 5

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 2 --ray-debugger-external --port 6380

MODEL_PATH=/data1/bowei/QUILL_ALL/model/Qwen2-7B-Instruct  # replace it with your local file path

CUDA_VISIBLE_DEVICES=0,2 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=2 \


    
