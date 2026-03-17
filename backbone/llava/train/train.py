import os
import sys
import subprocess
import pathlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common import (
    load_config,
    load_model_for_train,
    save_model,
    make_supervised_data_module,
)
from backbone.llava.train.llava_trainer import LLaVATrainer

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def train():
    global local_rank

    model_args, data_args, training_args = load_config()
    local_rank = training_args.local_rank
    model, tokenizer, data_args = load_model_for_train(model_args, data_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = LLaVATrainer(model=model,tokenizer=tokenizer,args=training_args,**data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print(f"Resuming from checkpoint in {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    rank0_print("Training completed, saving model...")

    save_model(model, training_args, trainer=trainer)

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("Cleaning intermediate checkpoints...")
        remove_dir = training_args.output_dir
        subprocess.run(
            f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +",
            shell=True
        )
        rank0_print("Done!")

if __name__ == "__main__":
    train()