# method/base/callback.py
from transformers import TrainerCallback


class CLTrainerCallback(TrainerCallback):
    """CL hooks on the training loop; no-op when the model is not a ``CLModel``."""

    def __init__(self, model_args, model):
        self.model_args = model_args
        self.integration = getattr(model, 'integration', None)
        self.model = model

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """After each step: optional prototype / state updates in ``on_step_end``."""
        if self.integration is None:
            return
        if model is None:
            model = self.model
        if hasattr(model, 'cl_context'):
            self.integration.on_step_end(model, model.cl_context, kwargs.get('loss'))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """End of training: optional ``on_train_task_finished`` then ``on_task_end``."""
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
