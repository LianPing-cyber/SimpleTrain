import subprocess
import threading
import os
class Trainer:
    def __init__(self, lf_path, lf_dir):
        self.start_file = lf_path + '/src/train.py'
        self.data_dir = lf_dir

    def get_source(self, model):
        self.model = model.output_model
        self.dataset = model.task_name

    def train(self, epoch):
        if self.stage == "sft":
            if self.finetuning_type == "full":
                self.train_sft_full(epoch)
            self.train_sft_lora(epoch)
    
    def train_sft_lora(self, epoch):
        cmd_head = []
        print(self.nproc_per_node)
        if int(self.nproc_per_node) < 2:
            cmd_head = ['python', '-u']
        else:
            cmd_head = ['torchrun', '--nproc_per_node', self.nproc_per_node, '--master_port', "29500"]
        cmd = cmd_head + [
            self.start_file,
            '--stage', 'sft',
            '--finetuning_type', 'lora',
            '--max_sample', self.max_sample,
            '--num_train_epochs', epoch,
            '--model_name_or_path', self.model,
            '--dataset', self.dataset,
            '--dataset_dir', self.data_dir,
            '--do_train', self.do_train,
            '--template', self.template,
            '--cutoff_len', self.cutoff_len,
            '--load_best_model_at_end', 'true',
            '--overwrite_cache', self.overwrite_cache,
            '--preprocessing_num_workers', self.preprocessing_num_workers,
            '--dataloader_num_workers', self.dataloader_num_workers,
            '--per_device_train_batch_size', self.per_device_train_batch_size,
            '--per_device_eval_batch_size', self.per_device_eval_batch_size,
            '--gradient_accumulation_steps', self.gradient_accumulation_steps,
            '--lr_scheduler_type', self.lr_scheduler_type,
            '--warmup_ratio', self.warmup_ratio,
            '--learning_rate', self.learning_rate,
            '--weight_decay', self.weight_decay,
            '--logging_steps', self.logging_steps,
            '--save_strategy', self.save_strategy,
            '--plot_loss', self.plot_loss,
            '--overwrite_output_dir', self.overwrite_output_dir,
            '--save_only_model', self.save_only_model,
            '--bf16', self.bf16,
            '--ddp_timeout', self.ddp_timeout
        ]
        adapter_path = os.path.join(self.model,"adp")
        if os.path.exists(adapter_path):
            cmd = cmd + ['--adapter_name_or_path', adapter_path]
        if self.finetuning_type == "lora":
            cmd = cmd + ['--lora_rank', self.lora_rank]
            cmd = cmd + ['--output_dir', os.path.join(self.model,"adp")]
        else:
            cmd = cmd + ['--output_dir', self.model]

        cmd = [str(arg) if not isinstance(arg, (str, bytes)) else arg for arg in cmd] 
        print("Start Train...")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1, 
            universal_newlines=True
        )
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "STDERR"))

        stdout_thread.start()
        stderr_thread.start()

        process.wait()

                
    def train_sft_full(self, epoch):
        pass

    def get_parameters(self, 
        stage,
        finetuning_type,
        max_sample,
        do_train="true",
        trust_remote_code="true",
        lora_rank="16",
        template="empty",
        cutoff_len="1024",
        overwrite_cache="true",
        preprocessing_num_workers="8",
        dataloader_num_workers="4",
        per_device_train_batch_size="2",
        per_device_eval_batch_size="4",
        gradient_accumulation_steps="4",
        lr_scheduler_type="cosine",
        warmup_ratio="0.1",
        learning_rate="5e-5",
        weight_decay="0.01",
        logging_steps="100",
        save_strategy="best",
        plot_loss="true",
        overwrite_output_dir="true",
        save_only_model="true",
        bf16="true",
        ddp_timeout="180000000",
        resume_from_checkpoint="null",
        nproc_per_node="2"
        ):
        self.stage = stage
        self.finetuning_type = finetuning_type
        self.max_sample = max_sample
        self.do_train = do_train
        self.trust_remote_code = trust_remote_code
        self.lora_rank = lora_rank
        self.template = template
        self.cutoff_len = cutoff_len
        self.overwrite_cache = overwrite_cache
        self.preprocessing_num_workers = preprocessing_num_workers
        self.dataloader_num_workers = dataloader_num_workers
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.plot_loss = plot_loss
        self.overwrite_output_dir = overwrite_output_dir
        self.save_only_model = save_only_model
        self.bf16 = bf16
        self.ddp_timeout = ddp_timeout
        self.resume_from_checkpoint = resume_from_checkpoint
        self.nproc_per_node = nproc_per_node

def read_output(stream, prefix):
    for line in iter(stream.readline, ''):
        print(f"{prefix}: {line}", end='')
    stream.close()