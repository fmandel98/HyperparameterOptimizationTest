from clearml import Task
from clearml.automation import DiscreteParameterRange, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer

task = Task.init(project_name='Fashion HPO bin',
                 task_name=f"optimizer",
                 task_type=Task.TaskTypes.optimizer)

optimizer = HyperParameterOptimizer(
    base_task_id='11eafb1744f84f74909ed8b8f88f16c3',
    hyper_parameters=[
        DiscreteParameterRange('General/optimizer', ['adam', 'sgd']),
        UniformIntegerParameterRange('General/hidden_dim', min_value=10, max_value=250),
        UniformParameterRange('General/dropout', min_value=0, max_value=0.4)
    ],
    objective_metric_title="epoch_accuracy",
    objective_metric_series='validation: epoch_accuracy',
    objective_metric_sign='max',
)

optimizer.start()
optimizer.wait()
optimizer.stop()