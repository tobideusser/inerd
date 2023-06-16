import argparse
import datetime
import logging
import os

import yaml
from fluidml import Flow
from fluidml.flow import TaskSpec

from misusing_llms import project_path
from misusing_llms.tasks import Parsing, Tokenisation, NERTraining, NEREvaluation, NERPreTraining
from misusing_llms.utils.fluid_helper import (
    configure_logging,
    MyLocalFileStore,
    TaskResource,
)
from misusing_llms.utils import get_balanced_devices, is_debug

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="Path to config",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Dataset name, used to load various configs, i.e. conll2003 loads the conll2003_config.yaml config. "
        "Ignored if --config is set",
    )
    parser.add_argument(
        "--cuda-ids",
        default=None,
        type=int,
        nargs="+",
        help="GPU ids, e.g. `--cuda-ids 0 1`",
    )
    parser.add_argument(
        "--cuda-group",
        default=None,
        type=int,
        help="How to group multiple GPU's, e.g. `--cuda-group 2` groups in pair of twos",
    )
    parser.add_argument("--deepspeed", action="store_true", help="Use deepspeed.")
    parser.add_argument("--fsdp", action="store_true", help="Use fsdp.")
    parser.add_argument("--pre-training", action="store_true", help="Execute only pre-training.")
    parser.add_argument("--use-cuda", action="store_true", help="Use cuda.")
    parser.add_argument("--max-split-size", type=int, default=None, help="Max. split size for cuda processes.")
    parser.add_argument("--warm-start", action="store_true", help="Tries to warm start training.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of multiprocessing workers.")
    parser.add_argument(
        "--force",
        type=str,
        nargs="+",
        default=None,
        help="Task or tasks to force execute. " + " registers successor tasks also for force execution."
        "E.g. --force ModelTraining+",
    )
    parser.add_argument(
        "--gs-expansion-method",
        type=str,
        default="product",
        choices=["product", "zip"],
        help="Method to expand config for grid search",
    )
    parser.add_argument("--log-to-tmux", action="store_true", help="Log to several tmux panes.")
    parser.add_argument("--project-name", type=str, default="iNERD", help="Name of project.")
    parser.add_argument("--run-name", type=str, default=None, help="Name of run.")
    return parser.parse_args()


def main():
    args = parse_args()

    do_pre_training = args.pre_training

    if args.config:
        config = yaml.safe_load(open(args.config, "r"))
    else:
        dataset = args.dataset
        if do_pre_training:
            config = yaml.safe_load(open(os.path.join(project_path, "scripts", "ner", "pretraining_config.yaml"), "r"))
        elif dataset == "conll2003":
            config = yaml.safe_load(open(os.path.join(project_path, "scripts", "ner", "conll2003_config.yaml"), "r"))
        elif dataset == "bc5cdr":
            config = yaml.safe_load(open(os.path.join(project_path, "scripts", "ner", "bc5cdr_config.yaml"), "r"))
        elif dataset == "ontonotes":
            config = yaml.safe_load(open(os.path.join(project_path, "scripts", "ner", "ontonotes_config.yaml"), "r"))
        else:
            raise ValueError(f"Dataset {dataset} not known.")

    base_dir = config["base_dir"]

    # Parse run settings from argparse (defaults and choices see above in argparse)
    num_workers = args.num_workers  # 1
    force = args.force  # "ModelTraining+"
    use_cuda = args.use_cuda
    cuda_ids = args.cuda_ids  # [1]  # [0, 1]
    cuda_group = args.cuda_group
    warm_start = args.warm_start  # False  # continue training from an existing checkpoint
    gs_expansion_method: str = args.gs_expansion_method
    run_name = "debug" if is_debug() else args.run_name
    project_name = args.project_name  # if args.dataset is None else args.project_name + "-" + args.dataset
    deepspeed = args.deepspeed
    fsdp = args.fsdp
    gpu_scaling = "auto"
    if deepspeed and fsdp:
        raise AssertionError("Two systems for gpu scaling specified (deepspeed and fsdp). Aborting.")
    elif deepspeed:
        gpu_scaling = "deepspeed"
    elif fsdp:
        gpu_scaling = "fsdp"

    # fixes pytorch memory leak
    # if use_cuda:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(cuda_id) for cuda_id in cuda_ids])
    #     cuda_ids = list(range(len(cuda_ids)))

    if args.max_split_size:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.max_split_size}"

    log_dir = os.path.join(base_dir, "logging")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(level="ERROR" if "LOCAL_RANK" in os.environ else "INFO", log_dir=log_dir)

    # get task configs
    data_parsing_cfg = config["Parsing"]
    tokenisation_cfg = config["Tokenisation"]
    pre_training_cfg = config["PreTraining"]

    # create task specs
    parsing = TaskSpec(task=Parsing, config=data_parsing_cfg)
    tokenisation = TaskSpec(task=Tokenisation, config=tokenisation_cfg)
    pre_training = TaskSpec(
        task=NERPreTraining,
        config=pre_training_cfg,
        expand=gs_expansion_method,
        additional_kwargs={"gpu_scaling": gpu_scaling},
    )

    # dependencies between tasks
    tokenisation.requires(parsing)
    pre_training.requires(tokenisation)

    if not do_pre_training:

        training_cfg = config["Training"]
        evaluation_cfg = config["Evaluation"]

        training = TaskSpec(task=NERTraining, config=training_cfg, expand=gs_expansion_method)
        evaluation = TaskSpec(task=NEREvaluation, config=evaluation_cfg, expand=gs_expansion_method)

        training.requires(pre_training, tokenisation)
        evaluation.requires(tokenisation, training)

        tasks = [
            parsing,
            tokenisation,
            pre_training,
            training,
            evaluation,
        ]

    else:

        tasks = [
            parsing,
            tokenisation,
            pre_training,
        ]

    # create list of resources
    devices = get_balanced_devices(count=num_workers, use_cuda=use_cuda, cuda_ids=cuda_ids, cuda_group=cuda_group)
    if devices == ["cpu"]:
        resources = [TaskResource(cuda=False, device="cpu") for _ in range(num_workers)]
    else:
        resources = [TaskResource(cuda=True, device=devices[i]) for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = MyLocalFileStore(base_dir=base_dir)

    start = datetime.datetime.now()
    # create flow (expanded task graph)
    flow = Flow(tasks=tasks)
    # run linearly without swarm if num_workers is set to 1
    # note resources are now assigned equally to all tasks (e.g. device info)
    # else run graph in parallel using multiprocessing
    # create list of resources which is distributed among workers
    # e.g. to manage that each worker has dedicated access to specific gpus
    flow.run(
        num_workers=args.num_workers,
        resources=resources,
        log_to_tmux=args.log_to_tmux,
        force=force,
        results_store=results_store,
        project_name=project_name,
        run_name=run_name,
    )

    end = datetime.datetime.now()
    logger.info(f"{end - start}")


if __name__ == "__main__":
    main()
