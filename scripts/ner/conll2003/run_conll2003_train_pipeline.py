import argparse
import datetime
import logging
import os

import yaml
from fluidml import Flow
from fluidml.flow import TaskSpec

from misusing_llms import project_path
from misusing_llms.tasks import Parsing, Tokenisation, NERTraining
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
        default=os.path.join(project_path, "scripts", "ner", "conll2003", "config.yaml"),
        type=str,
        help="Path to config",
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
    parser.add_argument("--project-name", type=str, default="misusing-llms", help="Name of project.")
    parser.add_argument("--run-name", type=str, default=None, help="Name of run.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = yaml.safe_load(open(args.config, "r"))

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
    training_cfg = config["Training"]

    # create all task specs
    parsing = TaskSpec(task=Parsing, config=data_parsing_cfg)
    tokenisation = TaskSpec(task=Tokenisation, config=tokenisation_cfg)
    # preprocessing = TaskSpec(task=Preprocessing, config=preprocessing_cfg)
    training = TaskSpec(
        task=NERTraining,
        config=training_cfg,
        expand=gs_expansion_method,
        # additional_kwargs=training_additional_kwargs,
    )

    # dependencies between tasks
    tokenisation.requires(parsing)
    # preprocessing.requires(tokenisation)
    training.requires(tokenisation)

    # list of all tasks
    tasks = [
        parsing,
        tokenisation,
        # preprocessing,
        training,
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
        project_name=args.project_name,
        run_name=run_name,
    )

    end = datetime.datetime.now()
    logger.info(f"{end - start}")


if __name__ == "__main__":
    main()
