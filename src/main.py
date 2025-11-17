from data_manager import Data_Normal
from model import Model
from train import Trainer
from eval import Evaluate_Normal

import torch
import argparse

import asyncio

def gpu_clear():
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Program Description")
    # Add command-line arguments
    parser.add_argument('--data', type=str, help='Input file path')
    parser.add_argument('--data_type', type=str, help='conversation or content', default='content')
    parser.add_argument('--task_name', type=str, help='Detailed output mode')
    parser.add_argument('--do_train', type=int, help="0 means no training, only evaluation; 1 means training + evaluation", default=0)
    parser.add_argument('--data_balance', type=int, help="1,2 to enable; 1 enables data_balance and will balance the data size of all classes every other training round (1,3...); 2 balances every time", default=1)
    parser.add_argument('--random_sample', type=int, help="1 to enable; with random_sample enabled, data selection for each round of training is random, otherwise, it's fixed random numbers", default=0)

    parser.add_argument('--epoch_num', type=int, help='Number of training epochs', default=1)
    parser.add_argument('--data_num', type=int, help='How much data you want to use for training', default=0)
    parser.add_argument('--eval_num', type=int, help='How much data you want to use for evaluation', default=0)

    parser.add_argument('--original_model_folder', type=str, help='Folder of the original model')
    parser.add_argument('--output_model_folder', type=str, help='Folder for saving the model')

    parser.add_argument('--lf_path',type=str, help='Llama_factory folder path, e.g., /root/llama_factory/')
    parser.add_argument('--lf_data_dir', type=str, help='Where to save llama_factory datasets')

    parser.add_argument('--per_device_train_batch_size', type=int, help='Batch size for each GPU', default=1)
    parser.add_argument('--nproc_per_node', type=int, help='Parallelization setting for number of processes per node during training', default=2)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=5e-5)

    args = parser.parse_args()

    data = args.data
    task_name = args.task_name
    data_type = args.data_type
    do_train = args.do_train
    data_balance = args.data_balance
    random_sample = args.random_sample

    epoch_num = args.epoch_num
    data_num = args.data_num
    eval_num = args.eval_num

    original_model = args.original_model_folder
    output_model = args.output_model_folder

    lf_path = args.lf_path
    lf_data_dir = args.lf_data_dir

    # Write a unified system prompt

    # First, load and save the dataset
    data = Data_Normal(data_name=task_name, data_path=data, data_type=data_type)
    if do_train == 1:
        data.get_data(max_num=data_num, random_sample=random_sample)
    data.get_eval_data(max_num=eval_num)

    if do_train == 1:
        trainer = Trainer(lf_path, lf_data_dir)
        trainer.get_parameters(stage="sft", finetuning_type="lora", max_sample=data_num,
                per_device_train_batch_size = args.per_device_train_batch_size,
                nproc_per_node=args.nproc_per_node,
                learning_rate=args.learning_rate
                )
        Qwen3Guard = Model(original_model)
        Qwen3Guard.new_train_task(output_model, task_name, trainer)

        for i in range(epoch_num):
            if data_balance == 2:
                data.data_balance()
            elif i%2 == 1 and data_balance == 1:
                data.data_balance()
            data.write_in_alpaca(save_file=lf_data_dir+"/"+task_name+".json")
            Qwen3Guard.train(1)
            gpu_clear()
            data.get_data(max_num=data_num, random_sample=random_sample)

    evaluater = Evaluate_Normal(output_model, data)
    evaluater.generate_result(eval_num)
    asyncio.run(evaluater.eval_right())

if __name__ == "__main__":
    main()
