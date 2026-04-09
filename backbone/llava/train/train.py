import os
import sys
import subprocess
import pathlib
# train.py - 仅需在原有基础上增加 2 行
# ... 原有 import 保持不变 ...
from common import (
    load_config,
    load_model_for_train,  # 这个函数内部会自动处理 CL 包装
    save_model,
    make_supervised_data_module,
)
from backbone.llava.train.llava_trainer import LLaVATrainer
# [新增] 导入 CL Callback（仅当使用 CL 方法时需要）
from method.base.callback import CLTrainerCallback  # noqa: F401

from pathlib import Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
_current_file = Path(__file__).absolute()
_project_root = _current_file.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    print(f"✅ [train.py] 已添加项目根目录：{_project_root}")


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
    trainer.add_callback(CLTrainerCallback(model_args, model))
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

