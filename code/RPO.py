import os
import json
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field, validator
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler


# ======================= CONFIGURATION =======================
class LoraConfigModel(BaseModel):
    """LoRA configuration parameters"""
    r: int = Field(..., gt=0, description="Rank of LoRA")
    lora_alpha: int = Field(..., gt=0, description="Alpha parameter of LoRA")
    lora_dropout: float = Field(0.0, ge=0, le=1, description="Dropout rate for LoRA layers")
    target_modules: List[str] = Field(..., description="Target modules to apply LoRA")


class ModelConfig(BaseModel):
    """Model related configuration"""
    path: str = Field(..., description="Pretrained model path")
    finetuning_type: Literal["lora", "full"] = Field(..., description="Fine-tuning type: lora or full")
    lora: Optional[LoraConfigModel] = Field(None, description="LoRA configuration")

    @validator('lora', always=True)
    def validate_lora(cls, v, values):
        """Validate that LoRA config must be provided when finetuning_type is 'lora'"""
        if values.get('finetuning_type') == 'lora' and v is None:
            raise ValueError("Lora config required when finetuning_type is 'lora'")
        return v


class TrainingConfig(BaseModel):
    """Training process configuration parameters"""
    lambda_val: float = Field(..., alias="lambda", gt=0, description="Lambda parameter in loss function")
    beta0: float = Field(..., gt=0, description="Initial beta parameter")
    s_max: int = Field(..., gt=0, description="Maximum score threshold")
    s_min: int = Field(0, ge=0, description="Minimum score threshold")
    per_device_train_batch_size: int = Field(..., gt=0, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(..., gt=0, description="Number of gradient accumulation steps")
    learning_rate: float = Field(..., gt=0, description="Learning rate")
    num_train_epochs: float = Field(..., gt=0, description="Number of training epochs")
    lr_scheduler_type: str = Field(..., description="Learning rate scheduler type")
    warmup_ratio: float = Field(0.0, ge=0, le=1, description="Warmup ratio")
    grad_clip: Optional[float] = Field(None, gt=0, description="Gradient clipping threshold")
    bf16: bool = Field(False, description="Whether to use bfloat16 precision")


class DataConfig(BaseModel):
    """Data related configuration"""
    dataset_path: str = Field(..., description="Dataset path")
    template: str = Field(..., description="Dialogue template type")
    cutoff_len: int = Field(..., gt=0, description="Maximum text length")
    max_samples: Optional[int] = Field(None, gt=0, description="Maximum sample limit")
    overwrite_cache: bool = Field(False, description="Whether to overwrite cache")
    preprocessing_num_workers: int = Field(0, ge=0, description="Number of preprocessing worker threads")


class OutputConfig(BaseModel):
    """Output related configuration"""
    output_dir: str = Field(..., description="Output directory")
    logging_steps: int = Field(..., gt=0, description="Logging step interval")
    save_steps: int = Field(..., gt=0, description="Model saving step interval")
    plot_loss: bool = Field(True, description="Whether to plot loss curve")


class FullConfig(BaseModel):
    """Full configuration structure"""
    model: ModelConfig = Field(..., description="Model configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    data: DataConfig = Field(..., description="Data configuration")
    output: OutputConfig = Field(..., description="Output configuration")


def load_config(config_path: str) -> FullConfig:
    """
    Load and validate the configuration file
    Args:
        config_path: path to the config file
    Returns:
        validated configuration object
    """
    with open(config_path) as f:
        config_data = yaml.safe_load(f)  # safe load YAML

    # Replace environment variables (supports ${VAR} and ${VAR:-default} format)
    config_data = replace_env_vars(config_data)
    return FullConfig(**config_data)


def replace_env_vars(data):
    """
    Recursively replace environment variables in config
    Args:
        data: config data (dict/list/str)
    Returns:
        data with env vars replaced
    """
    if isinstance(data, dict):
        return {k: replace_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
        var_name = data[2:-1].split(':-')
        if len(var_name) > 1:  # handle default value case ${VAR:-default}
            return os.getenv(var_name[0], var_name[1])
        return os.getenv(var_name[0], data)
    return data


# ======================= TEMPLATE PROCESSOR =======================
class QWenTemplateProcessor:
    """Qwen model dialogue template processor"""

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: tokenizer instance
        """
        self.tokenizer = tokenizer
        # system prompt template
        self.system_prompt = "<|im_start|>system\nYou are a roleplay expert assistant<|im_end|>\n"
        # user input template
        self.user_template = "<|im_start|>user\n{prompt}<|im_end|>\n"
        # assistant reply start marker
        self.assistant_start = "<|im_start|>assistant\n"

    def apply_template(self, prompt):
        """
        Apply dialogue template to user input
        Args:
            prompt: raw user input
        Returns:
            formatted dialogue text
        """
        return (
                self.system_prompt +
                self.user_template.format(prompt=prompt) +
                self.assistant_start
        )


# ======================= DATASET HANDLING =======================
class RoleplayDataset(Dataset):
    """Roleplay dialogue dataset"""

    def __init__(self, raw_data, processor, tokenizer, config):
        """
        Args:
            raw_data: raw data
            processor: template processor
            tokenizer: tokenizer
            config: configuration object
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.groups = self._process_data(raw_data)

    def _process_data(self, raw_data):
        """Process raw data to generate training sample pairs"""
        groups = []
        for item in tqdm(raw_data[:self.config.data.max_samples], desc="Processing data"):
            try:
                # apply dialogue template
                templated_prompt = self.processor.apply_template(item["prompt"])
                # filter and sort responses
                responses = sorted(
                    [r for r in item["responses"] if self._validate_response(r)],
                    key=lambda x: -x["score"]  # descending by score
                )
                # create response pairs
                pairs = self._create_response_pairs(responses)

                if pairs:
                    groups.append({
                        "prompt": templated_prompt,
                        "pairs": pairs
                    })
            except Exception as e:
                print(f"Error processing item: {str(e)}")
        return groups

    def _validate_response(self, response):
        """Validate if response is valid"""
        return (
                len(response["text"].strip()) > 0 and  # non-empty text
                response["score"] >= self.config.training.s_min and  # score not below min
                response["score"] <= self.config.training.s_max  # score not above max
        )

    def _create_response_pairs(self, responses):
        """Create training response pairs (high score vs low score)"""
        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if responses[i]["score"] > responses[j]["score"]:
                    pairs.append({
                        "y_sh": self._truncate(responses[i]["text"]),  # high-score response
                        "y_sl": self._truncate(responses[j]["text"]),  # low-score response
                        "s_h": responses[i]["score"],  # high score
                        "s_l": responses[j]["score"]  # low score
                    })
        return pairs

    def _truncate(self, text):
        """Truncate text to maximum length"""
        return self.tokenizer.decode(
            self.tokenizer.encode(text, truncation=True, max_length=self.config.data.cutoff_len)[:-1]
            # remove eos token
        )

    def __len__(self):
        """Return dataset size"""
        return len(self.groups)

    def __getitem__(self, idx):
        """Get a single sample"""
        return self.groups[idx]


# ======================= DATA LOADER =======================
class GroupedBatchSampler(BatchSampler):
    """Custom batch sampler that keeps samples within groups together"""

    def __init__(self, sampler, batch_size, drop_last=False):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        """Generate batch indices"""
        indices = list(range(len(self.sampler)))
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]


def load_dataset(config):
    """Load dataset"""
    dataset_path = Path(config.data.dataset_path).expanduser()  # expand ~
    if dataset_path.is_dir():
        # if directory, load all JSON files
        raw_data = []
        for fname in dataset_path.glob("*.json"):
            with open(fname, "r") as f:
                raw_data.extend(json.load(f))
        return raw_data
    else:
        # single file direct load
        with open(dataset_path, "r") as f:
            return json.load(f)


def create_dataloader(config, tokenizer):
    """Create data loader"""
    raw_data = load_dataset(config)
    processor = QWenTemplateProcessor(tokenizer)
    dataset = RoleplayDataset(raw_data, processor, tokenizer, config)

    return DataLoader(
        dataset,
        batch_sampler=GroupedBatchSampler(
            SequentialSampler(dataset),
            batch_size=config.training.per_device_train_batch_size,
            drop_last=False
        ),
        collate_fn=lambda b: collate_fn(b, tokenizer, config),
        num_workers=config.data.preprocessing_num_workers
    )


def collate_fn(batch, tokenizer, config):
    """Batch processing function"""
    processed_batch = []
    for group in batch:
        prompt = group["prompt"]
        for pair in group["pairs"]:
            # process high-score response
            sh_data = tokenize_response(prompt, pair["y_sh"], tokenizer, config)
            # process low-score response
            sl_data = tokenize_response(prompt, pair["y_sl"], tokenizer, config)

            processed_batch.append({
                "s_h": pair["s_h"],  # high score
                "s_l": pair["s_l"],  # low score
                "sh_inputs": sh_data,  # high-score inputs
                "sl_inputs": sl_data  # low-score inputs
            })
    return processed_batch


def tokenize_response(prompt, response, tokenizer, config):
    """Convert response to model inputs"""
    full_text = prompt + response
    inputs = tokenizer(
        full_text,
        max_length=config.data.cutoff_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    # create labels (ignore prompt part)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=config.data.cutoff_len)["input_ids"])
    labels = inputs.input_ids.clone()
    labels[:, :prompt_len] = -100  # ignore prompt part in loss computation
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels
    }


# ======================= TRAINER =======================
class HierarchicalDPOTrainer:
    """Hierarchical DPO trainer"""

    def __init__(self, config: FullConfig):
        """
        Args:
            config: full config object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0  # current training step
        self.total_loss = 0  # accumulated loss

        # initialize model and tokenizer
        self.model, self.tokenizer = self._init_model()
        # initialize reference model
        self.ref_model = self._init_ref_model()
        # initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._init_optimizer()

        # compute hierarchical weights
        s_h_values = list(range(config.training.s_min, config.training.s_max + 1))
        denominator = sum(2 ** s for s in s_h_values)
        self.w_sh = {s: (2 ** s) / denominator for s in s_h_values}

    def _init_model(self):
        """Initialize main model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float32,
        ).to(self.device)

        # enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # disable cache during training

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.path)
        tokenizer.pad_token = tokenizer.eos_token  # use eos token as pad token

        # apply LoRA config
        if self.config.model.finetuning_type == "lora":
            peft_config = LoraConfig(
                r=self.config.model.lora.r,
                lora_alpha=self.config.model.lora.lora_alpha,
                lora_dropout=self.config.model.lora.lora_dropout,
                target_modules=self.config.model.lora.target_modules
            )
            model = get_peft_model(model, peft_config)

        return model, tokenizer

    def _init_ref_model(self):
        """Initialize reference model (shares base weights with main model)"""
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float32,
        ).to(self.device)
        ref_model.eval()  # set reference model to eval mode
        return ref_model

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )

        # cosine learning rate scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.training.warmup_ratio * self.config.training.num_train_epochs * 100),
            num_training_steps=self.config.training.num_train_epochs * 100
        )

        return optimizer, scheduler

    def compute_logp(self, model, inputs):
        """Compute log probability"""
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode(model is self.ref_model):
            outputs = model(**inputs)
        return -outputs.loss  # negative loss equals log probability

    def hierarchical_loss(self, batch):
        """Calculate hierarchical DPO loss"""
        total_loss = 0
        for item in batch:
            # compute policy model logp for high-score response
            policy_sh_logp = self.compute_logp(self.model, item["sh_inputs"])
            # compute policy model logp for low-score response
            policy_sl_logp = self.compute_logp(self.model, item["sl_inputs"])

            # compute reference model logp for high-score response
            with torch.no_grad():
                ref_sh_logp = self.compute_logp(self.ref_model, item["sh_inputs"])
                # compute reference model logp for low-score response
                ref_sl_logp = self.compute_logp(self.ref_model, item["sl_inputs"])

            # compute differences between policy and reference logps
            diff = (policy_sh_logp - ref_sh_logp) - (policy_sl_logp - ref_sl_logp)

            # compute dynamic parameter
            delta = torch.tensor(
                (item["s_h"] - item["s_l"]) / (self.config.training.s_max - self.config.training.s_min)
            )
            beta_sh = self.config.training.beta0 * torch.exp(
                self.config.training.lambda_val * (1 - delta)
            )

            # compute loss term
            loss_term = -F.logsigmoid(beta_sh * delta * diff)
            # weighted sum of loss
            total_loss += self.w_sh[item["s_h"]] * loss_term

        return total_loss / len(batch)  # average loss

    def train_step(self, batch):
        """Perform a single training step"""
        self.model.train()
        # compute loss
        loss = self.hierarchical_loss(batch)

        # gradient accumulation
        loss = loss / self.config.training.gradient_accumulation_steps
        loss.backward()

        # optimization step
        if (self.step + 1) % self.config.training.gradient_accumulation_steps == 0:
            # gradient clipping
            if self.config.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # logging
        self.total_loss += loss.item()
        if self.step % self.config.output.logging_steps == 0:
            avg_loss = self.total_loss / self.config.output.logging_steps
            print(f"Step {self.step}: Loss={avg_loss:.4f}")
            self.total_loss = 0  # reset accumulated loss

        # save checkpoint
        if self.step % self.config.output.save_steps == 0:
            self.save_model()

        self.step += 1  # increment step
        return loss.item()

    def save_model(self):
        """Save model"""
        output_dir = Path(self.config.output.output_dir) / f"checkpoint-{self.step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.model.finetuning_type == "lora":
            # save LoRA weights
            self.model.save_pretrained(output_dir)
        else:
            # save full model
            torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")

        print(f"Model saved to {output_dir}")


# ======================= MAIN TRAINING =======================
def main():
    # disable PyTorch multiprocessing (avoid issues on some environments)
    os.environ["PYTORCH_ENABLE_MULTIPROCESSING"] = "0"

    # load configuration file
    config = load_config("config.yaml")

    # initialize trainer
    trainer = HierarchicalDPOTrainer(config)

    # create data loader
    dataloader = create_dataloader(config, trainer.tokenizer)

    # training loop
    for epoch in range(int(config.training.num_train_epochs)):
        # use tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # perform training step
            loss = trainer.train_step(batch)
            # update tqdm postfix
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

    # save final model
    trainer.save_model()


if __name__ == "__main__":
    main()