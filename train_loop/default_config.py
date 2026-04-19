from omegaconf import OmegaConf

DEFAULT_CONFIG = OmegaConf.create({
    "sanity": False,

    "model": {
        "name": None
    },

    "dataset": {
        "name": None
    },

    "losses": {
        "default_loss": {
            "function_name": None,
            "src_key": None ,
            "tgt_key": None 
        } ,
    },

    "download_assets" : {}, # To safely download theese assets before training starts

    "gpus": None,  # List of GPU ids to use (if None, use all available GPUs from env)

    "train": {
        "device": "cuda",

        "pt_single_threaded": True,
        "use_ddp": False,
        "n_gpus": 1,
        "gradient_accum_steps": 1,
        "find_unused_parameters": True,
        "compile_model": False,
        "is_compile_model_with_loss": False,
        "grad_clip_value": 0,
        "resume": False,
        "use_tqdm": False,
        "summary_frequency": 1000,
        "num_eval_every_steps": 1000,
        "training_outs_log_frequency" : 1000,
        "log_frequency": 1,
        "metrics_rolling_window": 1,
        "batch_size": 2, 
        "num_workers": 0,
        "checkpoint_save_frequency": 1000,
        "checkpoint_save_iterations": None,  # list of exact iterations to save at (in addition to frequency-based saves)

        "save_dir" : None, 

        "optimizer": {
            "name" : "adam"
        },

        "validation_use_tqdm": False,
        "num_validation_steps": -1,
        "create_model_meta_init": False,
        "use_bfloat16_autocast": False,
        "use_float16_autocast": False,
        "debug_load_batch_only_once": False,
        "debug_ddp_sync" : False,
        "no_save_weights": False,
        "save_separate_stepwise_checkpoints": False,
        "save_optimizer_in_all_checkpoints": False,
        "get_optimizer_fn_name": None,
        "resume_checkpoint_path": None,
        "resume_steps_num": None,
        "n_total_steps": -1 , 
        "n_total_samples" : -1 , 
        "log_iter_time" : False,
        "use_grad_scaler": False,
        "use_accelerate" : False,
        "accelerate_mixed_precision": None ,  # bf16 or fp16 or None
        "nccl_timeout_minutes": 60,

        "crash_detect_params" : None, 
        "crash_recovery_mode" : "resume_from_latest_checkpoint" # exit, resume_from_latest_checkpoint, reinit
    }
})