# method/base/callback.py
from transformers import TrainerCallback

class CLTrainerCallback(TrainerCallback):
    """
    持续学习训练回调
    如果模型不是 CLModel，所有方法直接跳过（零开销）
    """
    def __init__(self, model_args, model):
        self.model_args = model_args
        # 只有包装后的模型才有 integration
        self.integration = getattr(model, 'integration', None)
        self.model = model
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """每个训练步结束后：更新原型/状态"""
        if self.integration is None:
            return  # 非 CL 方法直接跳过
        if model is None:
            model = self.model
        if hasattr(model, 'cl_context'):
            self.integration.on_step_end(model, model.cl_context, kwargs.get('loss'))
            
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """训练结束时：可选 ``on_train_task_finished``（需 Fisher / trainer），再 ``on_task_end``。"""
        if self.integration is None:
            return
        if model is None:
            model = self.model
        if hasattr(model, "cl_context"):
            task_id = getattr(self.model_args, "cur_task", 0)
            hook = getattr(self.integration, "on_train_task_finished", None)
            if callable(hook):
                hook(
                    model,
                    model.cl_context,
                    task_id,
                    trainer=getattr(self, "trainer", None),
                )
            self.integration.on_task_end(model, model.cl_context, task_id)