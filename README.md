# 内部开发指南
Update by Jun-tao Tang, 2026/4/14

## Part 1. 如何运行（训练 / 推理 / 如何改配置参数）

### 1) 先配路径（必改）

训练/推理会通过 `--app-config {base|instruct}` 选择不同的路径配置：

- `config/paths/paths_config_instruct.py`
- `config/paths/paths_config_base.py`

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
python run.py train 0 --benchmark coin --method same --app-config instruct --gpus 0,1 --port 29601 --console

# 连续训练 task 0 和 task 1（task1 会自动带上 previous_task_model_path）
python run.py train 0 1 --benchmark coin --method same --app-config instruct --gpus 0,1 --port 29601 --console
```

日志默认写到 `output/train/<benchmark>/<method>/taskXX/run_*.txt`（加 `--console` 会同时输出到终端）。

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
python run.py train 1 --benchmark coin --method same --app-config instruct --gpus 0,1 --console \
  --debug \
  --port 29601
```

（更细的训练超参如 `--learning_rate/--warmup_ratio/--num_train_epochs/--per_device_train_batch_size` 等，建议放在 `config/methods/<method>.py` 里统一管理。）

### 4) 推理 / 评测入口

推理/评测同样通过：

```bash
python run.py infer 0 1 --benchmark coin --method same --app-config instruct --gpus 0,1 --console
```

> 说明：推理子命令还包含 `--model-path/--checkpoint-task/--checkpoint-suffix/--clmethod` 等参数，具体以 `run.py infer -h` 的输出为准。

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

4) **（推荐）实现方法额外状态保存/加载**

如果方法有“跨任务状态”（比如 anchors、Gram、covariance 快照等），实现：

- `save_extra_state(output_dir)`
- `load_extra_state(load_dir, model=...)`

训练时会在加载 `previous_task_model_path` 后自动调用 `load_extra_state()`（见 `common/load_model.py`）。

### B. 增加方法配置（强烈推荐）

新建 `config/methods/my_method.py`，至少提供：

- `TRAIN_FLAG_OVERRIDES`: 设置 `--method my_method`、LoRA/训练超参默认值
- `TRAIN_EXTRA_ARGS`: 需要额外透传给训练脚本的参数
- （可选）`TRAIN_BATCH_SIZES`: 每个 benchmark / task 的 batch size

这样你就可以直接：

```bash
python run.py train 0 1 --benchmark coin --method my_method --app-config instruct --gpus 0,1 --console
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
- 组装 config（r/alpha/dropout/expert_num/cur_task/target_modules 等）
- 对 `CLModel` 的 `_base_model` 调用 `get_peft_model(...)` 并回写 `_base_model`

---

## 常见坑（快速排雷）

- **类名反射失败**：方法目录叫 `my_method`，类名必须能被 `my_method.capitalize()` 找到（即 `My_methodIntegration`）。
- **方法额外状态没真的加载**：建议在 `load_extra_state()` 里做显式校验（例如关键 buffer/anchors 是否恢复），否则容易“打印加载成功但实际没生效”。
- **checkpoint 路径**：`run.py` 会把 checkpoint 组织成 `checkpoints/<Benchmark>/<method>/TaskX_llava_lora/`，不要手动改目录结构。