# 内部开发指南
Updated by tjt

## Part 1. 如何运行（训练 / 推理 / 如何改配置参数）

### 1) 配路径

路径统一在 **`config/paths/paths.py`** 中修改（不再区分 base / instruct）。

你需要把下面这些路径改成你机器上的真实位置：

- **`BASE_MODEL_PATH`**：LLaVA 基座模型目录（例如 `/root/autodl-tmp/LLaVa`）
- **`CLIP_PATH`**：CLIP/Text tower 路径（例如 `/root/autodl-tmp/CLIP`）
- **`MCIT_ROOT` / `INSTRUCTION_DIR` / `IMAGE_FOLDER`**：数据与图片根目录
- **`DEEPSPEED_CONFIG`**：deepspeed 配置（默认 `config/deepspeed/zero2.json`）
- **`CHECKPOINT_DIR` / `RESULT_DIR`**：输出目录

### 2) 训练入口

统一入口是 `run.py`（它会自动拼 deepspeed 命令、写日志、管理任务间 checkpoint 路径）。

常用命令：

```bash

注意，其实你可以直接 python run.py train 0，其他参数都可以在config/run_config里面改
# 训练 CoIN 的 task 0
python run.py train 0 --benchmark coin --method same --gpus 0,1 --port 29601

# 连续训练 task 0 和 task 1（task1 会自动带上 previous_task_model_path）
python run.py train 0 1 --benchmark coin --method same --gpus 0,1 --port 29601
```

日志默认写到 `output/<backbone>/train/<benchmark>/<method>/taskXX/run_*.txt`（`backbone` 见 `config/backbones/llava.py` 的 `BACKBONE_ID`）。

### 3) 如何改“训练参数/方法参数”

`run.py` 的训练参数来自两层配置叠加：

- **全局默认**：`config/run_config.py`（如果不存在会自动使用空默认）
- **方法默认**：`config/methods/<method>.py`

其中最常用的是 `TRAIN_FLAG_OVERRIDES`（覆盖命令行 flag）和 `TRAIN_EXTRA_ARGS`（追加额外参数）。

例如你想改 SAME 的 LoRA rank / 学习率 / batch size：

- 改 `config/methods/same.py` 里的 **`TRAIN_FLAG_OVERRIDES`**
- 或者直接在命令行加同名 flag（`run.py` 会做覆盖/追加）

示例（命令行覆盖）：

```bash
python run.py train 1 --benchmark coin --method same --gpus 0,1 \
  --debug \
  --port 29601
```

（更细的训练超参如 `--learning_rate/--warmup_ratio/--num_train_epochs/--per_device_train_batch_size` 等，建议放在 `config/methods/<method>.py` 里统一管理。）

### 4) 推理 / 评测入口

推理/评测同样通过：

```bash
python run.py infer 0 1 --benchmark coin --method same --gpus 0,1
```

> 说明：推理子命令还包含 `--model-path/--checkpoint-task/--checkpoint-suffix/--method`（传给评测子进程）等参数，具体以 `run.py infer -h` 的输出为准。评测结果目录为 `results/<backbone>/<Benchmark>/<method>/<任务名>/<stage>/`。

---

## Part 2. 如何增加一个新方法（要改哪些地方？含 PEFT 扩展）

新增一个方法（例如 `my_method`）通常分两块：**方法集成层（Integration）** + **（可选）自定义 PEFT tuner**。

### A. 增加方法集成层（必做）

1) **新建方法目录**

- 新建 `method/my_method/`
- 至少包含 `method/my_method/integration.py`

2) **实现 Integration 类**

在 `method/my_method/integration.py` 里实现：

- 类名必须是 **`My_methodIntegration`** 这种“首字母大写”的形式（`run.py` / `common/load_model.py` 会用 `f\"{method_name.capitalize()}Integration\"` 反射取类）
- 用装饰器注册方法名/别名：

```python
from method.factory import CLMethodFactory
from method.base.integration import CLIntegration

@CLMethodFactory.register("my_method", "my")
class My_methodIntegration(CLIntegration):
    ...
```

3) **接入生命周期钩子**

你至少要实现 `CLIntegration` 的抽象接口（见 `method/base/integration.py`）：

- `initialize_model()`: 冻结/解冻参数、注入 PEFT、初始化 router/anchors/原型等
- `on_input_prep()`: 输入阶段路由/状态更新
- `on_forward_start()/on_forward_end()`: forward 前后逻辑（可选）
- `on_step_end()/on_task_end()`: 训练步/任务结束时的更新与保存（可选）

4) **实现方法额外状态保存/加载**

如果方法有“跨任务状态”（比如 anchors、Gram、covariance 快照等），实现：

- `save_extra_state(output_dir)`
- `load_extra_state(load_dir, model=...)`

训练时会在加载 `previous_task_model_path` 后自动调用 `load_extra_state()`（见 `common/load_model.py`）。

### B. 增加方法配置

新建 `config/methods/my_method.py`，至少提供：

- `TRAIN_FLAG_OVERRIDES`: 设置 `--method my_method`、LoRA/训练超参默认值
- `TRAIN_EXTRA_ARGS`: 需要额外透传给训练脚本的参数
- （可选）`TRAIN_BATCH_SIZES`: 每个 benchmark / task 的 batch size

这样你就可以直接：

```bash
python run.py train 0 1 --benchmark coin --method my_method --gpus 0,1
```

### C. （可选）增加新的 PEFT 方法 / tuner

如果你的方法需要自定义 PEFT 类型（类似 `hide_llava` / `same`），需要做三件事：

1) **实现 tuner**

把 tuner 放到 `PEFT/peft/tuners/<your_tuner>.py`，包含：

- `Config`（dataclass，设置 `peft_type`）
- `Model/Layer`（如何把 LoRA/专家/router 注入进 Linear 等模块）

2) **注册到项目的 PEFT 扩展映射**

PEFT 侧需要把你的自定义 `peft_type/config/model` 注入到 PEFT 的 **mapping** 里，否则 `get_peft_model(...)` 不知道如何构造你的层。

本项目已经提供了运行时注入工具：`method/base/peft_extension.py::register_peft_extension()`，它会写入：

- `PEFT.peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls`
- `PEFT.peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = tuner_model_cls`（可选）
- `PEFT.peft.mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type] = task_peft_model_cls`（可选，用于自定义 PeftModel 包装类）

因此你需要在你的 `method/my_method/integration.py` 里提供一个**幂等**注册函数（参考 `method/same/integration.py`、`method/hide_llava/integration.py`）：

```python
from method.base.peft_extension import register_peft_extension

def ensure_peft_extension_registered():
    from PEFT.peft.tuners.your_tuner import YourConfig, YourModel
    register_peft_extension(
        peft_type="YOUR_PEFT_TYPE",
        config_cls=YourConfig,
        tuner_model_cls=YourModel,
        # 如果你需要自定义的 PeftModel 包装类，也可以额外传：
        # task_type="CAUSAL_LM_XXX",
        # task_peft_model_cls=YourPeftModelForCausalLM,
    )
```

然后在 `initialize_model()` 里在 `get_peft_model(...)` 之前调用它，确保映射存在。

3) **在 Integration 里注入**

在 `initialize_model()` 中：

- 冻结 backbone 参数（如需要）
- 组装 config（r/alpha/dropout/cur_task/target_modules 等；`task_num` 由 `--benchmark` 与 `config/benchmarks` 中的任务数对齐；PEFT 层仍可能使用 `expert_num` 字段名以兼容权重）
- 对 `CLModel` 的 `_base_model` 调用 `get_peft_model(...)` 并回写 `_base_model`

---
