## Informed Named Entity Decoding for Generative Language Models

This is the repository for the **iNERD** ("Informed Named Entity Recognition Decoding for Generative Language Models") 
paper. It contains all code necessary to reproduce the results of this paper.

### iNERD algorithm

If you simply want to use the iNERD algorithm head to ``inerd/training/logits_processor.py``. The class 
``InformedNERDecoderLogitsProcessor`` implements the Algorithm 1 (iNERD) described in the paper.

### Training pipeline

To run the training and evaluation pipeline, you should first install all required packages by running 
``pip install /path/to/setup.py``, preferably in a virtual environment. Afterwards, pick a dataset that you want to 
train on and adjust the config file for that dataset (located in ``scripts/ner``) to suit your requirements. 

Following this, simply run the script ``scripts/ner/run_ner_training_pipeline.py`` with the following parameters of 
your choice:

- `--config`: If you want to load a different config, overwrites the ``--dataset`` argument.
- `--dataset`: Dataset name, used to load various configs, i.e. conll2003 loads the conll2003_config.yaml config. Ignored if ``--config`` is set.
- `--use-cuda`: Set to `true` to run the pipeline with GPU support. Automatically set to `true` if `--cuda-ids` is specified.
- `--cuda-ids`: Which GPU ids to use, e.g. `--cuda-ids 0 1` used GPU 0 & 1.
- `--cuda-group`: How to group multiple GPU's, e.g. `--cuda-group 2` groups in pairs of twos.
- `--deepspeed`: Use the deepspeed backend for multi-GPU computation.
- `--fsdp`: Use the fsdp backend for multi-GPU computation
- `--pre-training`: Execute everything up to and including the pre-training step (coarse tuning in the paper)
- `--no-pre-training`: Run the training pipeline without additional pre-training (coarse tuning in the paper)
- `--no-zeroshot`: Do not run a zero-shot evaluation before fine-tuning.
- `--warm-start`: If set, tries to warm start the training, i.e. load from a previous checkpoint.
- `--num-workers`: Number of fluidML multiprocessing workers.
- `--force`: Task or tasks to force execute.  '+'  registers successor tasks also for force execution. E.g. --force ModelTraining+
- `--gs-expansion-method`: Method to expand config for grid search. `choices=["product", "zip"]`
- `--checkpointing-time-interval`: Time interval to do model checkpointing - format DD:hh:mm
- `--run-name`: Name of run, shown in folder structure and wandb.

#### Example

```
python scripts/ner/run_ner_training_pipeline.py --dataset conlpp --cuda-ids 0
```

-> Runs the complete training pipeline for the CoNLL++ dataset with GPU 0 as an accelerator.

#### Using Llama

To be able to load the Llama model the pre-trained weights are required. You can apply at this website for the weights: 
https://ai.meta.com/blog/large-language-model-llama-meta-ai/

After getting the weights, run the `scripts/convert_llama_weights_to_hf.py` script to convert the llama weights to the 
HuggingFace format to be able to load them. Then, simply paste the absolute path to these converted weights into the 
config under `tokeniser_name: &modelname FILE_PATH_HERE` and set the `llama` parameter to `true`.

#### Hardware requirements

To run the training pipeline smoothly, we recommend a GPU with at least 40 GB of VRAM for the models with a size below 7
billion parameters and a GPU with 80 GB for the Llama 13 billion version.
