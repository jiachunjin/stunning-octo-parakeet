def get_accelerator(config):
    import os
    import pprint
    from omegaconf import OmegaConf
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    from accelerate.utils import ProjectConfiguration

    output_dir = os.path.join(config.train.root, config.train.exp_name, config.train.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.train.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.train.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with                    = config.train.report_to,
        mixed_precision             = config.train.mixed_precision,
        project_config              = project_config,
        gradient_accumulation_steps = config.train.gradient_accumulation_steps,
    )
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    return accelerator, output_dir