import os
import sys
import torch
import torchaudio
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, Audio  # type: ignore
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
from functools import partial
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import load_file as load_safetensors

# Add the 'src' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from chatterbox.models.s3gen.s3gen import S3Token2Mel, S3Token2Wav
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR as S3_TOKENIZER_SR
from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer, SPEECH_VOCAB_SIZE, ModelConfig as S3TokenizerModelConfig
from chatterbox.models.s3gen.xvector import CAMPPlus
from chatterbox.models.s3gen.utils.mel import mel_spectrogram
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor

# New imports for T3 (text→S3)
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

# ========================
# Dataset (unchanged, but now includes 'text')
# ========================
class TTSDataset(Dataset):
    def __init__(self, data_source, audio_column='audio', text_column='transcript', audio_base_path=None,
                 target_sr_for_s3_tokenizer=S3_TOKENIZER_SR, target_sr_for_mel=S3GEN_SR):
        self.data_source = data_source
        self.audio_column = audio_column
        self.text_column = text_column
        self.audio_base_path = audio_base_path
        self.target_sr_for_s3_tokenizer = target_sr_for_s3_tokenizer
        self.target_sr_for_mel = target_sr_for_mel

        if hasattr(self.data_source, 'column_names') and self.audio_column not in self.data_source.column_names:
            for col_name in self.data_source.column_names:
                try:
                    sample_item = self.data_source[0][col_name]
                    if isinstance(sample_item, dict) and 'array' in sample_item and 'sampling_rate' in sample_item:
                        self.audio_column = col_name
                        break
                except Exception:
                    continue

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        sample = self.data_source[idx]
        audio_info = sample[self.audio_column]
        transcription_raw = sample[self.text_column]
        transcription = str(transcription_raw) if transcription_raw is not None else ""

        if isinstance(audio_info, dict) and 'array' in audio_info:
            waveform_array = audio_info['array']
            if waveform_array is None:
                raise ValueError(f"Audio array is None for sample at index {idx}, path/id: {audio_info.get('path', 'N/A')}")
            waveform = torch.tensor(waveform_array, dtype=torch.float32)
            original_sr = audio_info['sampling_rate']
            audio_path = audio_info.get('path', f"hf_item_{idx}")
        else:
            path_str = str(audio_info)
            resolved_path = os.path.join(self.audio_base_path or '', path_str)
            if not os.path.exists(resolved_path):
                current_script_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_str)
                if self.audio_base_path is None and os.path.exists(current_script_dir_path):
                    resolved_path = current_script_dir_path
                elif isinstance(self.data_source, list) and hasattr(sample, '__csv_file_path__'):
                    csv_dir = os.path.dirname(sample['__csv_file_path__'])
                    alt_path_csv = os.path.join(csv_dir, path_str)
                    if os.path.exists(alt_path_csv):
                        resolved_path = alt_path_csv
                elif hasattr(self.data_source, 'csv_file_path') and isinstance(self.data_source.csv_file_path, str):
                    csv_dir = os.path.dirname(self.data_source.csv_file_path)
                    alt_path_df = os.path.join(csv_dir, path_str)
                    if os.path.exists(alt_path_df):
                        resolved_path = alt_path_df
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"Audio file not found: {path_str} (final attempt: {resolved_path})")
            waveform, original_sr = torchaudio.load(resolved_path)  # type: ignore
            audio_path = resolved_path

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.numel() == 0:
            raise ValueError(f"Empty waveform loaded for sample {idx}, path: {audio_path}. Original SR: {original_sr}")

        if original_sr != self.target_sr_for_s3_tokenizer:
            resampler_16k = torchaudio.transforms.Resample(original_sr, self.target_sr_for_s3_tokenizer)
            waveform_16k = resampler_16k(waveform)
        else:
            waveform_16k = waveform

        if original_sr != self.target_sr_for_mel:
            resampler_24k = torchaudio.transforms.Resample(original_sr, self.target_sr_for_mel)
            waveform_24k = resampler_24k(waveform)
        else:
            waveform_24k = waveform

        if waveform_16k.numel() == 0 or waveform_24k.numel() == 0:
            raise ValueError(f"Empty waveform after resampling for sample {idx}, path: {audio_path}. Original SR: {original_sr}, 16k len: {waveform_16k.numel()}, 24k len: {waveform_24k.numel()}")

        return {
            'waveform_16k': waveform_16k,
            'waveform_24k': waveform_24k,
            'text': transcription,
            'audio_path': audio_path,
            'original_sr': original_sr
        }


# ========================
# Worker Initialization Function (unchanged)
# ========================
worker_s3_tokenizer_instance: S3Tokenizer
worker_mel_extractor_target_fn = None
worker_speaker_encoder_instance: CAMPPlus
worker_padding_s3_token_id: float = 0.0
worker_token_mel_ratio: int = 2
worker_collate_device: torch.device


def custom_worker_init_fn(worker_id,
                          s3_tokenizer_name, s3_tokenizer_config_args,
                          s3_tokenizer_state_dict,
                          speaker_encoder_state_dict,
                          campplus_feat_dim, campplus_embedding_size,
                          padding_id, token_mel_ratio, device_str,
                          mel_extractor_target_s3gen
                          ):
    global worker_s3_tokenizer_instance, worker_mel_extractor_target_fn
    global worker_speaker_encoder_instance, worker_padding_s3_token_id
    global worker_token_mel_ratio, worker_collate_device

    worker_collate_device = torch.device(device_str)
    worker_padding_s3_token_id = padding_id
    worker_token_mel_ratio = token_mel_ratio
    worker_mel_extractor_target_fn = mel_extractor_target_s3gen

    s3_config_for_worker = S3TokenizerModelConfig(**s3_tokenizer_config_args) if isinstance(
        s3_tokenizer_config_args, dict) else S3TokenizerModelConfig()

    worker_s3_tokenizer_instance = S3Tokenizer(name=s3_tokenizer_name, config=s3_config_for_worker)
    if s3_tokenizer_state_dict:
        worker_s3_tokenizer_instance.load_state_dict(s3_tokenizer_state_dict)
    worker_s3_tokenizer_instance.to(worker_collate_device)
    worker_s3_tokenizer_instance.eval()

    worker_speaker_encoder_instance = CAMPPlus(feat_dim=campplus_feat_dim, embedding_size=campplus_embedding_size)
    worker_speaker_encoder_instance.load_state_dict(speaker_encoder_state_dict)
    worker_speaker_encoder_instance.to(worker_collate_device)
    worker_speaker_encoder_instance.eval()


# ========================
# Collate Function (now includes 'texts' and 'raw_waveforms_16k_list')
# ========================
class TTSCollate:
    def __init__(self, device_for_collate_main_thread: torch.device, token_mel_r: int):
        self.device_main_thread = device_for_collate_main_thread
        self.token_mel_ratio_main_thread = token_mel_r

    def __call__(self, batch):
        current_s3_tokenizer: S3Tokenizer
        current_mel_extractor_target: callable
        current_speaker_encoder: CAMPPlus
        current_padding_id: float
        current_token_mel_ratio: int
        current_device_to_use: torch.device

        try:
            _ = worker_s3_tokenizer_instance
            current_s3_tokenizer = worker_s3_tokenizer_instance
            current_mel_extractor_target = worker_mel_extractor_target_fn  # type: ignore
            current_speaker_encoder = worker_speaker_encoder_instance
            current_padding_id = worker_padding_s3_token_id
            current_token_mel_ratio = worker_token_mel_ratio
            current_device_to_use = worker_collate_device
        except NameError:
            current_s3_tokenizer = g_s3_tokenizer_instance_main  # type: ignore
            current_mel_extractor_target = g_mel_extractor_target_fn_main  # type: ignore
            current_speaker_encoder = g_speaker_encoder_instance_main  # type: ignore
            current_padding_id = g_padding_s3_token_id_main  # type: ignore
            current_token_mel_ratio = g_token_mel_ratio_main  # type: ignore
            current_device_to_use = self.device_main_thread

        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        texts = [item['text'] for item in batch]
        waveforms_16k_list = [item['waveform_16k'] for item in batch]
        waveforms_24k_list = [item['waveform_24k'] for item in batch]

        processed_speech_tokens = []
        processed_speech_feat = []
        processed_embeddings = []
        speech_token_lens_list = []
        speech_feat_lens_list = []

        for i, (wf16, wf24) in enumerate(zip(waveforms_16k_list, waveforms_24k_list)):
            if wf16.numel() == 0 or wf24.numel() == 0:
                continue
            wf16_dev = wf16.to(current_device_to_use)
            wf24_dev = wf24.to(current_device_to_use)

            s3_tokens_b, s3_len_b = current_s3_tokenizer(wf16_dev.unsqueeze(0))
            processed_speech_tokens.append(s3_tokens_b.squeeze(0))
            speech_token_lens_list.append(s3_len_b.item())

            mel_feat_b = current_mel_extractor_target(
                y=wf24_dev.unsqueeze(0),
                n_fft=1920, num_mels=80, sampling_rate=S3GEN_SR, hop_size=480, win_size=1920,
                fmin=0, fmax=8000, center=False
            ).transpose(1, 2)
            processed_speech_feat.append(mel_feat_b.squeeze(0))
            speech_feat_lens_list.append(mel_feat_b.size(1))

            with torch.no_grad():
                current_speaker_encoder.eval()
                embedding_b = current_speaker_encoder.inference([wf16_dev])
            processed_embeddings.append(embedding_b.squeeze(0))

        if not processed_speech_tokens:
            return None

        speech_tokens_padded = pad_sequence(processed_speech_tokens, batch_first=True,
                                            padding_value=current_padding_id).to(current_device_to_use)
        speech_feat_padded = pad_sequence(processed_speech_feat, batch_first=True,
                                          padding_value=0.0).to(current_device_to_use)

        embeddings_stacked = torch.stack(processed_embeddings).to(current_device_to_use)
        speech_token_lens_tensor = torch.tensor(speech_token_lens_list, dtype=torch.long).to(current_device_to_use)
        speech_feat_lens_tensor = torch.tensor(speech_feat_lens_list, dtype=torch.long).to(current_device_to_use)

        max_feat_len_batch = speech_feat_padded.size(1)
        if max_feat_len_batch > 0 and max_feat_len_batch % current_token_mel_ratio != 0:
            target_max_feat_len = (max_feat_len_batch // current_token_mel_ratio) * current_token_mel_ratio
            if target_max_feat_len > 0:
                speech_feat_padded = speech_feat_padded[:, :target_max_feat_len, :]
                speech_feat_lens_tensor = torch.clamp(speech_feat_lens_tensor, max=target_max_feat_len)

        max_mel_len_adjusted = speech_feat_padded.size(1)
        expected_max_s3_len = max_mel_len_adjusted // current_token_mel_ratio

        if expected_max_s3_len <= 0 and max_mel_len_adjusted > 0:
            if max_mel_len_adjusted < current_token_mel_ratio:
                return None
            else:
                expected_max_s3_len = 1

        if speech_tokens_padded.size(1) > expected_max_s3_len and expected_max_s3_len > 0:
            speech_tokens_padded = speech_tokens_padded[:, :expected_max_s3_len]
            speech_token_lens_tensor = torch.clamp(speech_token_lens_tensor, max=expected_max_s3_len)
        elif speech_tokens_padded.size(1) < expected_max_s3_len:
            if expected_max_s3_len > 0:
                diff = expected_max_s3_len - speech_tokens_padded.size(1)
                if diff > 0:
                    padding_tensor = torch.full((speech_tokens_padded.size(0), diff), current_padding_id,
                                                dtype=speech_tokens_padded.dtype, device=current_device_to_use)
                    speech_tokens_padded = torch.cat([speech_tokens_padded, padding_tensor], dim=1)

        if speech_tokens_padded.size(1) == 0 and speech_feat_padded.size(1) > 0:
            return None

        final_s3_len = speech_tokens_padded.size(1)
        final_mel_len = speech_feat_padded.size(1)

        if final_s3_len > 0 and final_mel_len != final_s3_len * current_token_mel_ratio:
            target_mel_len_from_s3 = final_s3_len * current_token_mel_ratio
            if target_mel_len_from_s3 <= 0:
                return None

            if final_mel_len > target_mel_len_from_s3:
                speech_feat_padded = speech_feat_padded[:, :target_mel_len_from_s3, :]
                speech_feat_lens_tensor = torch.clamp(speech_feat_lens_tensor, max=target_mel_len_from_s3)
            elif final_mel_len < target_mel_len_from_s3:
                diff_mel = target_mel_len_from_s3 - final_mel_len
                mel_padding = torch.zeros((speech_feat_padded.size(0), diff_mel, speech_feat_padded.size(2)),
                                          dtype=speech_feat_padded.dtype, device=current_device_to_use)
                speech_feat_padded = torch.cat([speech_feat_padded, mel_padding], dim=1)

        if speech_tokens_padded.numel() == 0:
            return None

        return {
            'speech_token': speech_tokens_padded.long(),
            'speech_token_len': speech_token_lens_tensor,
            'speech_feat': speech_feat_padded,
            'speech_feat_len': speech_feat_lens_tensor,
            'embedding': embeddings_stacked,
            'texts': texts,
            'raw_waveforms_16k_list': waveforms_16k_list,
        }


# ========================
# CLI Arguments (unchanged)
# ========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Chatterbox TTS (S3Token2Mel + T3) model")
    parser.add_argument('--csv_path', type=str, help="Path to training CSV file.")
    parser.add_argument('--csv_eval_path', type=str, help="Path to evaluation CSV file.")
    parser.add_argument('--hf_dataset', type=str, help="HuggingFace dataset name")
    parser.add_argument('--hf_dataset_config', type=str, default=None, help="HuggingFace dataset configuration")
    parser.add_argument('--hf_dataset_split_train', type=str, default="train",
                        help="HuggingFace dataset split for training.")
    parser.add_argument('--hf_dataset_split_eval', type=str, default="dev",
                        help="HuggingFace dataset split for evaluation. Use 'none' to disable.")
    parser.add_argument('--audio_column', type=str, default='audio', help="Name of the audio column in CSV/HF.")
    parser.add_argument('--text_column', type=str, default='transcript', help="Name of the text/transcription column.")
    parser.add_argument('--audio_base_path', type=str, default=None, help="Base path for audio files if using CSV.")
    parser.add_argument('--output_dir', type=str, default='./chatterbox_finetune_checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Base learning rate.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./chatterbox_finetune_logs')
    parser.add_argument('--log_interval', type=int, default=100, help="Log training loss every N steps.")
    parser.add_argument('--eval_interval', type=int, default=1, help="Run evaluation every N epochs.")
    parser.add_argument('--sample_interval', type=int, default=500,
                        help="Log audio sample every N global steps.")
    parser.add_argument('--save_interval', type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze_layers', type=str, nargs='*', default=[],
                        help="List of layer name patterns to freeze.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_audio_len_s', type=float, default=15.0, help="Max audio length in seconds.")
    parser.add_argument('--finetune_checkpoint_path', type=str, default=None,
                        help="Path to a fine-tuning checkpoint to resume training from.")
    parser.add_argument('--pretrained_s3gen_path', type=str, default=None,
                        help="Path to the PRE-TRAINED s3gen.pt (or .safetensors) from ResembleAI/chatterbox for initial weights.")
    parser.add_argument('--num_warmup_steps', type=int, default=1000,
                        help="Number of linear warmup steps for learning rate.")
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help="Maximum number of training samples to use (for quick testing).")
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help="Maximum number of evaluation samples to use (for quick testing).")
    return parser.parse_args()


# ========================
# Freeze layers (unchanged)
# ========================
def freeze_layers_fn(model, layer_names_to_freeze):
    if not layer_names_to_freeze:
        for param in model.parameters():
            param.requires_grad = True
        return
    print("Applying layer freezing...")
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        for layer_pattern in layer_names_to_freeze:
            if layer_pattern in name:
                param.requires_grad = False
                break


# ========================
# Prepare HF Dataset (unchanged)
# ========================
def prepare_hf_dataset_fn(dataset_name, config_name, split_name, audio_col, text_col, max_audio_len_s,
                          target_sr_tokenizer, num_workers_filter):
    print(f"Loading HuggingFace dataset: {dataset_name}, config: {config_name}, split: {split_name}")
    ds = load_dataset(dataset_name, name=config_name, split=split_name, trust_remote_code=True)
    try:
        ds = ds.cast_column(audio_col, Audio())
    except Exception as e:
        print(f"[WARN] Failed to cast audio column '{audio_col}' with Audio(): {e}.")

    max_samples_at_tokenizer_sr = int(max_audio_len_s * target_sr_tokenizer)

    def filter_examples(example):
        try:
            if example[text_col] is None or not str(example[text_col]).strip():
                return False
            audio_data = example[audio_col]
            if audio_data is None or audio_data['array'] is None:
                return False
            original_sr = audio_data['sampling_rate']
            if original_sr is None or original_sr <= 0:
                return False
            num_samples_original = len(audio_data['array'])
            if num_samples_original == 0:
                return False
            num_samples_at_tokenizer_sr_est = int(num_samples_original * (target_sr_tokenizer / original_sr))
            min_samples_at_tokenizer_sr = int(0.5 * target_sr_tokenizer)
            return min_samples_at_tokenizer_sr <= num_samples_at_tokenizer_sr_est <= max_samples_at_tokenizer_sr
        except Exception:
            return False

    original_len = len(ds)
    actual_num_proc_filter = num_workers_filter if num_workers_filter is not None and num_workers_filter > 0 else None

    print(f"Filtering dataset (max_len_s={max_audio_len_s}, min_len_s=0.5) with num_proc={actual_num_proc_filter}...")
    ds_filtered = ds.filter(filter_examples, num_proc=actual_num_proc_filter)
    filtered_len = len(ds_filtered)
    print(f"Filtered dataset from {original_len} to {filtered_len} samples.")
    if filtered_len == 0 and original_len > 0:
        print("[WARN] Dataset is empty after filtering. Returning original dataset (this may lead to errors).")
        return ds
    elif filtered_len == 0 and original_len == 0:
        raise ValueError("Dataset is empty BEFORE filtering. Check dataset name, config, or split.")
    return ds_filtered


# ========================
# Load Checkpoint (now includes T3 state)
# ========================
def load_checkpoint_fn(checkpoint_path, s3mel_model, t3_model, optimizer, scheduler, device):
    start_epoch = 0
    global_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading fine-tuning checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            # Load S3Token2Mel
            new_model_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_model_state_dict[name] = v
            s3mel_model.load_state_dict(new_model_state_dict)

            # Load T3
            if 't3_state_dict' in checkpoint:
                new_t3_state = OrderedDict()
                for k, v in checkpoint['t3_state_dict'].items():
                    name = k[7:] if k.startswith('module.') else k
                    new_t3_state[name] = v
                t3_model.load_state_dict(new_t3_state)
            else:
                print("[WARN] Checkpoint does not contain 't3_state_dict'. Training T3 from scratch.")

            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                if isinstance(scheduler, type(torch.optim.lr_scheduler.ReduceLROnPlateau())) and 'factor' in checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                elif hasattr(scheduler, 'load_state_dict') and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR) and 'last_epoch' in checkpoint['scheduler_state_dict']:  # type: ignore
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    print("Scheduler type in checkpoint might be different or state incompatible, not loading scheduler state.")

            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            print(f"Resuming fine-tuning from epoch {start_epoch + 1}, global_step {global_step}")
        except RuntimeError as e:
            print(f"Error loading state_dict from fine-tuning checkpoint: {e}. Attempting to load with strict=False...")
            try:
                s3mel_model.load_state_dict(new_model_state_dict, strict=False)
                print("Loaded S3Token2Mel with strict=False.")
                if 't3_state_dict' in checkpoint:
                    t3_model.load_state_dict(new_t3_state, strict=False)
                    print("Loaded T3 with strict=False.")
                start_epoch = checkpoint.get('epoch', 0)
                global_step = checkpoint.get('global_step', 0)
            except Exception as e2:
                print(f"Could not load models from checkpoint even with strict=False: {e2}.")
    return start_epoch, global_step


# ###################################
# MAIN BLOCK
# ###################################
g_s3_tokenizer_instance_main: S3Tokenizer
g_mel_extractor_target_fn_main = None
g_speaker_encoder_instance_main: CAMPPlus
g_padding_s3_token_id_main: float = 0.0
g_token_mel_ratio_main: int = 2

if __name__ == '__main__':
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':  # type: ignore
            mp.set_start_method('spawn', force=True)  # type: ignore
            print("Multiprocessing start method set to 'spawn'.")
    except Exception as e:
        if "context has already been set" not in str(e).lower() and "cannot start a new process before" not in str(e).lower():
            print(f"Warning: Could not set start method to 'spawn': {e}")

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    current_device = torch.device(args.device)

    # ========================
    # 1) Instantiate S3Token2Mel and Vocoder
    # ========================
    s3mel_model_finetune = S3Token2Mel()
    vocoder_for_sampling: HiFTGenerator

    if args.pretrained_s3gen_path and os.path.exists(args.pretrained_s3gen_path):
        print(f"Loading ALL components from OFFICIAL pre-trained checkpoint: {args.pretrained_s3gen_path}")
        temp_s3token2wav_model = S3Token2Wav()
        try:
            if args.pretrained_s3gen_path.endswith(".safetensors"):
                state_dict_from_file = load_safetensors(args.pretrained_s3gen_path, device='cpu')
            else:
                ckpt_content = torch.load(args.pretrained_s3gen_path, map_location='cpu')
                if isinstance(ckpt_content, dict):
                    potential_keys = ['model_state_dict', 'state_dict', 'model']
                    state_dict_from_file = None
                    for key in potential_keys:
                        if key in ckpt_content:
                            if isinstance(ckpt_content[key], dict):
                                state_dict_from_file = ckpt_content[key]
                                break
                            elif hasattr(ckpt_content[key], 'state_dict'):
                                state_dict_from_file = ckpt_content[key].state_dict()
                                break
                    if state_dict_from_file is None:
                        state_dict_from_file = ckpt_content
                else:
                    state_dict_from_file = ckpt_content

            if not state_dict_from_file:
                raise ValueError("Could not extract a valid state_dict from the checkpoint file.")

            processed_state_dict = OrderedDict()
            has_module_prefix_overall = any(k.startswith('module.') for k in state_dict_from_file.keys())
            for k, v in state_dict_from_file.items():
                name = k[7:] if has_module_prefix_overall and k.startswith('module.') else k
                processed_state_dict[name] = v

            missing_keys_wav, unexpected_keys_wav = temp_s3token2wav_model.load_state_dict(processed_state_dict, strict=False)
            if missing_keys_wav:
                print(f"[WARN] Pretrained Load: Missing keys in S3Token2Wav: {missing_keys_wav}")
            if unexpected_keys_wav:
                print(f"[WARN] Pretrained Load: Unexpected keys in S3Token2Wav: {unexpected_keys_wav}")
            print("Weights loaded into temporary S3Token2Wav model.")

            print("Transferring weights to S3Token2Mel instance for fine-tuning...")
            s3mel_model_finetune.tokenizer.load_state_dict(temp_s3token2wav_model.tokenizer.state_dict())
            s3mel_model_finetune.speaker_encoder.load_state_dict(temp_s3token2wav_model.speaker_encoder.state_dict())
            s3mel_model_finetune.flow.load_state_dict(temp_s3token2wav_model.flow.state_dict())

            print("Weights for tokenizer, speaker_encoder, and flow module transferred to S3Token2Mel instance.")

            vocoder_for_sampling = temp_s3token2wav_model.mel2wav
            print("Pre-trained vocoder (mel2wav) obtained from S3Token2Wav instance.")
            del temp_s3token2wav_model
        except Exception as e:
            print(f"Error loading/transferring pre-trained weights from {args.pretrained_s3gen_path}: {e}")
            import traceback; traceback.print_exc()
            print("Proceeding with uninitialized S3Token2Mel. Vocoder will also be uninitialized.")
            s3mel_model_finetune = S3Token2Mel()
            vocoder_for_sampling = HiFTGenerator(f0_predictor=ConvRNNF0Predictor())
    else:
        print(f"No pre-trained S3Gen checkpoint provided. Initializing S3Token2Mel and Vocoder from scratch.")
        s3mel_model_finetune = S3Token2Mel()
        vocoder_for_sampling = HiFTGenerator(f0_predictor=ConvRNNF0Predictor())

    s3mel_model_finetune.to(current_device)
    vocoder_for_sampling.to(current_device)
    vocoder_for_sampling.eval()

    # ========================
    # 2) Load T3 and VoiceEncoder and EnTokenizer for text→S3
    # ========================
    print("Loading T3, VoiceEncoder, and EnTokenizer for text→S3 path...")
    # Download/define paths for T3 and VE safetensors and tokenizer JSON
    # You can replace these with hf_hub_download if preferred
    from huggingface_hub import hf_hub_download
    tokenizer_json_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="tokenizer.json")
    t3_safetensors_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="t3_cfg.safetensors")
    ve_safetensors_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="ve.safetensors")

    # 2.1) Load EnTokenizer
    text_tokenizer = EnTokenizer(tokenizer_json_path)

    # 2.2) Load T3
    t3_model = T3()
    sd_t3 = load_safetensors(t3_safetensors_path, device='cpu')
    processed_t3 = OrderedDict()
    has_mod_t3 = any(k.startswith('module.') for k in sd_t3.keys())
    for k, v in sd_t3.items():
        name = k[7:] if (has_mod_t3 and k.startswith('module.')) else k
        processed_t3[name] = v
    t3_model.load_state_dict(processed_t3)
    t3_model.to(current_device)
    t3_model.eval()

    # 2.3) Load VoiceEncoder
    ve_model = VoiceEncoder()
    sd_ve = load_safetensors(ve_safetensors_path, device='cpu')
    ve_model.load_state_dict(sd_ve)
    ve_model.to(current_device)
    ve_model.eval()

    # ========================
    # 3) Configure optimizer to include T3 (if you want to fine-tune T3)
    # ========================
    # Include parameters from S3Token2Mel and T3
    params_to_opt = list(filter(lambda p: p.requires_grad, s3mel_model_finetune.parameters()))
    params_to_opt += list(filter(lambda p: p.requires_grad, t3_model.parameters()))
    optimizer = optim.AdamW(params_to_opt, lr=args.learning_rate)
    scheduler = None

    # ========================
    # 4) Load checkpoint (including T3) if it exists
    # ========================
    start_epoch, global_step = load_checkpoint_fn(args.finetune_checkpoint_path,
                                                 s3mel_model_finetune,
                                                 t3_model,
                                                 optimizer,
                                                 scheduler,
                                                 current_device)

    if args.freeze_layers:
        freeze_layers_fn(s3mel_model_finetune, args.freeze_layers)
        # If you want to freeze parts of T3, do it here:
        # freeze_layers_fn(t3_model, ["encoder", "decoder.interesting_layer"])

    # Print trainable parameters
    s3_trainable = sum(p.numel() for p in s3mel_model_finetune.parameters() if p.requires_grad)
    s3_total = sum(p.numel() for p in s3mel_model_finetune.parameters())
    t3_trainable = sum(p.numel() for p in t3_model.parameters() if p.requires_grad)
    t3_total = sum(p.numel() for p in t3_model.parameters())
    print(f"S3Token2Mel - Trainable params: {s3_trainable} / {s3_total} ({s3_trainable / s3_total * 100:.2f}%)")
    print(f"T3           - Trainable params: {t3_trainable} / {t3_total} ({t3_trainable / t3_total * 100:.2f}%)")

    # ========================
    # 5) Prepare arguments for collate and instantiate DataLoader
    # ========================
    # (Same logic to extract tokenizer from workers)
    s3_tokenizer_name_for_worker = getattr(s3mel_model_finetune.tokenizer, 'name', "speech_tokenizer_v2_25hz")
    if not isinstance(s3_tokenizer_name_for_worker, str):
        s3_tokenizer_name_for_worker = "speech_tokenizer_v2_25hz"

    s3_tokenizer_config_value_from_model = getattr(s3mel_model_finetune.tokenizer, 'config', None)
    s3_tokenizer_config_for_worker_args = {}
    if s3_tokenizer_config_value_from_model is not None:
        try:
            if isinstance(s3_tokenizer_config_value_from_model, DictConfig):
                s3_tokenizer_config_for_worker_args = OmegaConf.to_container(
                    s3_tokenizer_config_value_from_model, resolve=True)  # type: ignore
            elif isinstance(s3_tokenizer_config_value_from_model, dict):
                s3_tokenizer_config_for_worker_args = s3_tokenizer_config_value_from_model
            elif hasattr(s3_tokenizer_config_value_from_model, '__dict__') and not callable(
                    s3_tokenizer_config_value_from_model):
                temp_dict = vars(s3_tokenizer_config_value_from_model).copy()
                s3_tokenizer_config_for_worker_args = {k: v for k, v in temp_dict.items() if not k.startswith('_')}
                print(f"[INFO] Converted s3mel_model.tokenizer.config (type: {type(s3_tokenizer_config_value_from_model)}) to dict.")
            else:
                print(
                    f"[WARN] s3mel_model.tokenizer.config (type: {type(s3_tokenizer_config_value_from_model)}) is unhandled.")
        except Exception as e:
            print(f"[WARN] Error converting s3_tokenizer_config: {e}. Using empty dict.")
    if not isinstance(s3_tokenizer_config_for_worker_args, dict):
        s3_tokenizer_config_for_worker_args = {}

    campplus_feat_dim_default = 80
    campplus_embedding_size_default = 192

    worker_init_args = {
        "s3_tokenizer_name": s3_tokenizer_name_for_worker,
        "s3_tokenizer_config_args": s3_tokenizer_config_for_worker_args,
        "s3_tokenizer_state_dict": s3mel_model_finetune.tokenizer.state_dict(),
        "speaker_encoder_state_dict": s3mel_model_finetune.speaker_encoder.state_dict(),
        "campplus_feat_dim": campplus_feat_dim_default,
        "campplus_embedding_size": campplus_embedding_size_default,
        "padding_id": float(SPEECH_VOCAB_SIZE + 1),
        "token_mel_ratio": s3mel_model_finetune.flow.token_mel_ratio,
        "device_str": args.device,
        "mel_extractor_target_s3gen": mel_spectrogram
    }

    print(f"Using padding_id = {worker_init_args['padding_id']} for S3 tokens in collate.")
    print(f"Using token_mel_ratio = {worker_init_args['token_mel_ratio']} in collate.")

    g_s3_tokenizer_instance_main = S3Tokenizer(
        name=worker_init_args["s3_tokenizer_name"],
        config=S3TokenizerModelConfig(**worker_init_args["s3_tokenizer_config_args"])
    )
    g_s3_tokenizer_instance_main.load_state_dict(worker_init_args["s3_tokenizer_state_dict"])
    g_s3_tokenizer_instance_main.to(current_device)
    g_s3_tokenizer_instance_main.eval()
    g_mel_extractor_target_fn_main = worker_init_args["mel_extractor_target_s3gen"]
    g_speaker_encoder_instance_main = CAMPPlus(feat_dim=worker_init_args["campplus_feat_dim"],
                                               embedding_size=worker_init_args["campplus_embedding_size"])
    g_speaker_encoder_instance_main.load_state_dict(worker_init_args["speaker_encoder_state_dict"])
    g_speaker_encoder_instance_main.to(current_device)
    g_speaker_encoder_instance_main.eval()
    g_padding_s3_token_id_main = worker_init_args["padding_id"]
    g_token_mel_ratio_main = worker_init_args["token_mel_ratio"]

    num_workers_for_filter = args.num_workers if args.num_workers > 0 and args.num_workers <= (
                mp.cpu_count() // 2 if mp.cpu_count() > 1 else 1) else (1 if args.num_workers > 0 else None)

    dataset_obj = None
    eval_dataset_obj = None

    if args.hf_dataset:
        train_ds_hf_full = prepare_hf_dataset_fn(
            args.hf_dataset, args.hf_dataset_config, args.hf_dataset_split_train,
            args.audio_column, args.text_column, args.max_audio_len_s, S3_TOKENIZER_SR, num_workers_for_filter
        )
        if args.max_train_samples is not None and args.max_train_samples < len(train_ds_hf_full):
            print(f"Using a subset of {args.max_train_samples} training samples (original: {len(train_ds_hf_full)}).")
            train_ds_hf = train_ds_hf_full.select(range(args.max_train_samples))
        else:
            train_ds_hf = train_ds_hf_full
        dataset_obj = TTSDataset(train_ds_hf, audio_column=args.audio_column, text_column=args.text_column)

        if args.hf_dataset_split_eval and args.hf_dataset_split_eval.lower() != 'none':
            try:
                eval_split_to_try = args.hf_dataset_split_eval
                if args.hf_dataset == "freds0/cml_tts_dataset_portuguese" and eval_split_to_try.lower() == "validation":
                    eval_split_to_try = "dev"
                if args.hf_dataset == "AdrienB134/portuguese-tts" and eval_split_to_try.lower() == "validation":
                    eval_split_to_try = "test"
                    print(f"For AdrienB134/portuguese-tts, attempting 'test' split for evaluation instead of 'validation'.")

                eval_ds_hf_full = prepare_hf_dataset_fn(
                    args.hf_dataset, args.hf_dataset_config, eval_split_to_try,
                    args.audio_column, args.text_column, args.max_audio_len_s, S3_TOKENIZER_SR, num_workers_for_filter
                )
                if args.max_eval_samples is not None and args.max_eval_samples < len(eval_ds_hf_full):
                    print(f"Using a subset of {args.max_eval_samples} evaluation samples (original: {len(eval_ds_hf_full)}).")
                    eval_ds_hf = eval_ds_hf_full.select(range(args.max_eval_samples))
                else:
                    eval_ds_hf = eval_ds_hf_full
                if len(eval_ds_hf) > 0:
                    eval_dataset_obj = TTSDataset(eval_ds_hf, audio_column=args.audio_column, text_column=args.text_column)
                    print(f"Successfully loaded evaluation split: {eval_split_to_try} (using {len(eval_ds_hf)} samples).")
                else:
                    print(f"Evaluation split {eval_split_to_try} is empty after filtering/selection. Eval skipped.")
            except Exception as e:
                print(f"Could not load evaluation split '{args.hf_dataset_split_eval}': {e}. Eval skipped.")
    elif args.csv_path:
        print(f"Loading CSV dataset from: {args.csv_path}")
        df_train_full = pd.read_csv(args.csv_path)
        records_train_full = df_train_full.to_dict(orient='records')
        for r_idx, r_train in enumerate(records_train_full):
            r_train['__csv_file_path__'] = args.csv_path
        if args.max_train_samples is not None and args.max_train_samples < len(records_train_full):
            print(f"Using a subset of {args.max_train_samples} training samples (original: {len(records_train_full)}).")
            records_train = records_train_full[:args.max_train_samples]
        else:
            records_train = records_train_full
        dataset_obj = TTSDataset(records_train, audio_column=args.audio_column, text_column=args.text_column,
                                 audio_base_path=args.audio_base_path)

        if args.csv_eval_path:
            df_eval_full = pd.read_csv(args.csv_eval_path)
            records_eval_full = df_eval_full.to_dict(orient='records')
            for r_idx, r_eval in enumerate(records_eval_full):
                r_eval['__csv_file_path__'] = args.csv_eval_path
            if args.max_eval_samples is not None and args.max_eval_samples < len(records_eval_full):
                print(f"Using a subset of {args.max_eval_samples} evaluation samples (original: {len(records_eval_full)}).")
                records_eval = records_eval_full[:args.max_eval_samples]
            else:
                records_eval = records_eval_full
            if len(records_eval) > 0:
                eval_dataset_obj = TTSDataset(records_eval, audio_column=args.audio_column, text_column=args.text_column,
                                              audio_base_path=args.audio_base_path)
                print(f"Successfully loaded CSV evaluation data (using {len(records_eval)} samples).")
            else:
                print(f"CSV evaluation data is empty after selection. Eval skipped.")
    else:
        raise ValueError("You must provide either --csv_path or --hf_dataset")

    if dataset_obj is None:
        raise ValueError("Training dataset could not be loaded. Exiting.")

    # ========================
    # 6) Scheduler Warmup
    # ========================
    if args.num_warmup_steps > 0:
        try:
            from transformers import get_linear_schedule_with_warmup

            num_optimizer_steps_per_epoch = len(dataset_obj) // (args.batch_size * args.gradient_accumulation_steps)
            if len(dataset_obj) % (args.batch_size * args.gradient_accumulation_steps) != 0:
                num_optimizer_steps_per_epoch += 1

            num_training_steps = (args.epochs - start_epoch) * num_optimizer_steps_per_epoch
            actual_warmup_steps = args.num_warmup_steps
            if num_training_steps <= args.num_warmup_steps and num_training_steps > 0:
                actual_warmup_steps = max(0, int(num_training_steps * 0.1))
                print(f"[WARN] num_warmup_steps ({args.num_warmup_steps}) >= num_training_steps ({num_training_steps}). Adjusting warmup to {actual_warmup_steps}.")
            elif num_training_steps <= 0:
                actual_warmup_steps = 0
                num_training_steps = 1
                print("[WARN] num_training_steps is <= 0. Setting warmup to 0 and total steps to 1 for scheduler.")

            if scheduler is None or not (start_epoch > 0 and args.finetune_checkpoint_path):
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=actual_warmup_steps,
                                                            num_training_steps=num_training_steps)
                print(f"Using NEW learning rate scheduler with {actual_warmup_steps} warmup steps and estimated {num_training_steps} total training steps.")
            else:
                print("Scheduler state was loaded from fine-tuning checkpoint (or should have been).")

        except ImportError:
            print("`transformers` library not found. Falling back to ReduceLROnPlateau.")
            if scheduler is None:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
                print("Using NEW ReduceLROnPlateau learning rate scheduler.")
            else:
                print("Scheduler (likely ReduceLROnPlateau) loaded from checkpoint.")
    else:
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            print("Using NEW ReduceLROnPlateau learning rate scheduler.")
        else:
            print("Scheduler was loaded from checkpoint.")

    # ========================
    # 7) DataLoader and Collate
    # ========================
    collate_fn_instance = TTSCollate(device_for_collate_main_thread=current_device,
                                     token_mel_r=g_token_mel_ratio_main)

    dataloader_num_workers = args.num_workers
    if mp.get_start_method(allow_none=True) != 'spawn' and args.device == 'cuda' and args.num_workers > 0:  # type: ignore
        print(f"[WARN] CUDA with num_workers > 0 and non-'spawn' method. Forcing num_workers = 0.")
        dataloader_num_workers = 0

    custom_worker_init_fn_partial = partial(custom_worker_init_fn, **worker_init_args)

    dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=True,
                            num_workers=dataloader_num_workers, collate_fn=collate_fn_instance,
                            pin_memory=True if args.device == "cuda" and dataloader_num_workers > 0 else False,
                            worker_init_fn=custom_worker_init_fn_partial if dataloader_num_workers > 0 else None,
                            persistent_workers=True if dataloader_num_workers > 0 else False,
                            prefetch_factor=2 if dataloader_num_workers > 0 else None)

    eval_loader = None
    if eval_dataset_obj and len(eval_dataset_obj) > 0:
        eval_loader = DataLoader(eval_dataset_obj, batch_size=1, shuffle=False,
                                 num_workers=dataloader_num_workers, collate_fn=collate_fn_instance,
                                 pin_memory=True if args.device == "cuda" and dataloader_num_workers > 0 else False,
                                 worker_init_fn=custom_worker_init_fn_partial if dataloader_num_workers > 0 else None,
                                 persistent_workers=True if dataloader_num_workers > 0 else False)

    # ========================
    # 8) Start training loop
    # ========================
    s3mel_model_finetune.train()
    t3_model.train()  # If we are fine-tuning T3 as well
    print(f"Starting training from epoch {start_epoch + 1} for {args.epochs} total epochs on {current_device}...")

    for epoch in range(start_epoch, args.epochs):
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        epoch_loss_sum = 0.0
        num_samples_processed_epoch = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        if args.gradient_accumulation_steps == 1:
            optimizer.zero_grad()

        for i, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue

            actual_batch_size = batch_data['speech_token'].size(0)
            if args.gradient_accumulation_steps > 1 and (i % args.gradient_accumulation_steps == 0):
                optimizer.zero_grad()

            try:
                # ======== (A) Reconstruction loss (tokens extracted from the audio itself) ========
                loss_dict = s3mel_model_finetune.flow.forward(batch_data, device=current_device)
                loss_recon = loss_dict['loss']

                # ======== (B) Text→mel loss ========
                texts_list = batch_data['texts']  # list of strings, size B
                wav16_list = batch_data['raw_waveforms_16k_list']  # list of 1-D tensors [T_waveform]

                # (B.1) Tokenize texts
                text_ids_list = []
                for txt in texts_list:
                    text_norm = txt.strip()
                    if len(text_norm) == 0:
                        text_norm = " "
                    if text_norm[0].islower():
                        text_norm = text_norm[0].upper() + text_norm[1:]
                    # Convert to IDs
                    text_ids = text_tokenizer.text_to_tokens(text_norm).to(current_device)
                    sot, eot = t3_model.hp.start_text_token, t3_model.hp.stop_text_token
                    text_ids = torch.nn.functional.pad(text_ids, (1, 0), value=sot)
                    text_ids = torch.nn.functional.pad(text_ids, (0, 1), value=eot)
                    text_ids_list.append(text_ids)

                max_len_text = max(t.size(0) for t in text_ids_list)
                text_ids_padded = torch.zeros((len(text_ids_list), max_len_text), dtype=torch.long,
                                              device=current_device)
                for idx_t, t_ids in enumerate(text_ids_list):
                    text_ids_padded[idx_t, :t_ids.size(0)] = t_ids

                # (B.2) Extract S3 prompts and speaker embeddings via VoiceEncoder
                #       to build T3Cond
                prompt_s3_list = []
                for wav16 in wav16_list:
                    # wav16: 1-D tensor [T_waveform], convert to [1, T_waveform]
                    wf = wav16.unsqueeze(0).to(current_device) if wav16.ndim == 1 else wav16.to(current_device)
                    with torch.no_grad():
                        s3_prompt, _ = s3mel_model_finetune.tokenizer(wf)
                    prompt_s3_list.append(s3_prompt.squeeze(0))

                t3_cond_prompt_len = t3_model.hp.speech_cond_prompt_len
                cond_prompts_batch = []
                for s3_prompt in prompt_s3_list:
                    if s3_prompt.size(0) > t3_cond_prompt_len:
                        cond_prompts_batch.append(s3_prompt[:t3_cond_prompt_len])
                    else:
                        pad_val = SPEECH_VOCAB_SIZE + 1
                        pad_size = t3_cond_prompt_len - s3_prompt.size(0)
                        padded = torch.cat([s3_prompt, torch.full((pad_size,), pad_val,
                                                                   device=current_device, dtype=torch.long)], dim=0)
                        cond_prompts_batch.append(padded)
                cond_prompts_batch = torch.stack(cond_prompts_batch, dim=0)  # (B, t3_cond_prompt_len)

                spk_emb_list = []
                for wav16 in wav16_list:
                    wf16 = wav16.unsqueeze(0).to(current_device) if wav16.ndim == 1 else wav16.to(current_device)
                    wav_np = wf16.squeeze(0).cpu().numpy()
                    emb_np = ve_model.embeds_from_wavs([wav_np], sample_rate=16000)[0]
                    emb_t = torch.from_numpy(emb_np).unsqueeze(0).to(current_device)
                    spk_emb_list.append(emb_t)
                spk_emb_batch = torch.cat(spk_emb_list, dim=0)  # (B, VE_DIM)

                t3_conds = T3Cond(
                    speaker_emb=spk_emb_batch,
                    cond_prompt_speech_tokens=cond_prompts_batch,
                    emotion_adv=0.5 * torch.ones((len(texts_list), 1, 1), device=current_device)
                )

                # (B.3) Generate S3 tokens via T3 (batch with CFG)
                text_ids_cfg = torch.cat([text_ids_padded, text_ids_padded], dim=0)  # (2B, max_len_text)
                if hasattr(t3_conds, 'repeat_batch') and callable(t3_conds.repeat_batch):
                    t3_conds_batched = t3_conds.repeat_batch(2)
                else:
                    kwargs = {}
                    kwargs['speaker_emb'] = t3_conds.speaker_emb.repeat(2, 1, 1)
                    kwargs['cond_prompt_speech_tokens'] = t3_conds.cond_prompt_speech_tokens.repeat(2, 1)
                    kwargs['emotion_adv'] = t3_conds.emotion_adv.repeat(2, 1, 1)
                    t3_conds_batched = T3Cond(**kwargs)

                with torch.no_grad():  # Uncomment if you want to train T3, remove no_grad
                    gen_s3_b = t3_model.inference(
                        t3_cond=t3_conds_batched,
                        text_tokens=text_ids_cfg,
                        max_new_tokens=1000,
                        temperature=0.7,
                        cfg_weight=0.5
                    )
                gen_s3 = gen_s3_b[:len(text_ids_list)]  # (B, T_s3_gen)
                s3_tokens_from_text = gen_s3.to(current_device)

                # (B.4) Generate mel_text from s3_tokens_from_text + ref_dict
                # Recompute ref_dict for each wav16, stack into batch
                ref_dicts = [s3mel_model_finetune.embed_ref(ref_wav=wav16.unsqueeze(0).to(current_device),
                                                            ref_sr=16000,
                                                            device=current_device) for wav16 in wav16_list]
                # Pack ref_dicts into a batch-aware structure
                # Assuming each ref_dict has keys 'prompt_token', 'speaker_emb', etc.
                # We build a batch_ref_dict with stacked tensors
                batch_ref_dict = {}
                for key in ref_dicts[0].keys():
                    # stack along dimension 0
                    batch_ref_dict[key] = torch.stack([rd[key] for rd in ref_dicts], dim=0)

                # Prepare the dictionary for textual forward
                textual_batch_data = {
                    'speech_token': s3_tokens_from_text.long(),                           # (B, T_s3_gen)
                    'speech_token_len': torch.tensor([s3_tokens_from_text.size(1)] * len(texts_list),
                                                    dtype=torch.long, device=current_device),
                    'embedding': spk_emb_batch,                                           # (B, VE_DIM)
                    'speech_feat': batch_data['speech_feat'],                              # (B, T_mel, 80)
                    'speech_feat_len': batch_data['speech_feat_len'],                      # (B,)
                    'ref_dict': batch_ref_dict                                              # batched dictionary
                }

                # Call flow.forward in textual mode (if there is a dedicated method) or reuse flow.forward
                # Assuming use of flow.forward, but with textual_batch_data
                loss_dict_text = s3mel_model_finetune.flow.forward(textual_batch_data, device=current_device)
                loss_text2mel = loss_dict_text['loss']

                # ======== (C) Sum losses ========
                lambda_text = 1.0
                loss_total = loss_recon + lambda_text * loss_text2mel
                actual_loss_for_backward = loss_total
                if args.gradient_accumulation_steps > 1:
                    actual_loss_for_backward = loss_total / args.gradient_accumulation_steps
                actual_loss_for_backward.backward()

                # ======== Update ========
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,
                                                          list(s3mel_model_finetune.parameters()) +
                                                          list(t3_model.parameters())), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None and args.num_warmup_steps > 0 and isinstance(scheduler,
                                                                                          torch.optim.lr_scheduler.LambdaLR):
                        scheduler.step()
                    optimizer.zero_grad()

                epoch_loss_sum += loss_total.item() * actual_batch_size
                num_samples_processed_epoch += actual_batch_size

                if global_step % args.log_interval == 0:
                    writer.add_scalar("Loss/train_step_recon", loss_recon.item(), global_step)
                    writer.add_scalar("Loss/train_step_text", loss_text2mel.item(), global_step)
                    writer.add_scalar("Loss/train_step_total", loss_total.item(), global_step)
                    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
                    progress_bar.set_postfix({"loss_total": f"{loss_total.item():.4f}",
                                              "lr": f"{optimizer.param_groups[0]['lr']:.1e}"})
            except Exception as e:
                print(f"[ERROR] training step {global_step}: {e}")
                import traceback; traceback.print_exc()
                if "CUDA out of memory" in str(e) and args.device == "cuda":
                    torch.cuda.empty_cache()  # type: ignore
                optimizer.zero_grad()
                continue

            # ======== Sample interval (same as original code) ========
            if global_step > 0 and global_step % args.sample_interval == 0:
                s3mel_model_finetune.eval()
                t3_model.eval()
                with torch.no_grad():
                    try:
                        idx_ = 0
                        if batch_data is None or not all(k in batch_data for k in ['raw_waveforms_16k_list', 'texts']):
                            continue
                        if idx_ >= len(batch_data['texts']):
                            continue

                        ref_wav_for_sample = batch_data['raw_waveforms_16k_list'][idx_].unsqueeze(0).to(current_device)
                        text_for_sample = batch_data['texts'][idx_]
                        target_mel_for_vocoder_input = batch_data['speech_feat'][idx_:idx_ + 1].to(current_device) if 'speech_feat' in batch_data and \
                                                                                                                        batch_data[                                                                                                                            'speech_feat'] is not None else None

                        current_ref_dict = s3mel_model_finetune.embed_ref(ref_wav=ref_wav_for_sample, ref_sr=S3_TOKENIZER_SR,
                                                                          device=current_device)

                        # (i) Pure reconstruction (using ref→tokens→mel)
                        tokens_for_generation = current_ref_dict['prompt_token']
                        if tokens_for_generation is None or tokens_for_generation.numel() == 0:
                            continue

                        gen_mels_ = s3mel_model_finetune.forward(
                            speech_tokens=tokens_for_generation,
                            ref_wav=None,
                            ref_sr=None,
                            ref_dict=current_ref_dict,
                            finalize=True
                        )
                        gen_audio_, _ = vocoder_for_sampling.inference(speech_feat=gen_mels_)
                        writer.add_audio(f"Sample/Recon_Audio_E{epoch + 1}_S{global_step}",
                                         gen_audio_.squeeze().cpu(), global_step, sample_rate=S3GEN_SR)
                        writer.add_text(f"Sample/Recon_Text_E{epoch + 1}_S{global_step}",
                                        text_for_sample, global_step)

                        # (ii) Text→TTS sample
                        # Build T3Cond and generate S3 tokens for sample idx_
                        sample_wav16 = batch_data['raw_waveforms_16k_list'][idx_].unsqueeze(0).to(current_device)
                        with torch.no_grad():
                            s3_prompt_sample, _ = s3mel_model_finetune.tokenizer(sample_wav16)
                            wav_np_sample = sample_wav16.squeeze(0).cpu().numpy()
                            emb_np_sample = ve_model.embeds_from_wavs([wav_np_sample], sample_rate=16000)[0]
                            ve_emb_sample = torch.from_numpy(emb_np_sample).unsqueeze(0).to(current_device)

                        # Normalize sample text
                        def punc_norm(text: str) -> str:
                            if len(text) == 0:
                                return "You need to add some text for me to talk."
                            if text[0].islower():
                                text = text[0].upper() + text[1:]
                            text = " ".join(text.split())
                            punc_to_replace = [
                                ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
                                ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""),
                                ("‘", "'"), ("’", "'"),
                            ]
                            for old, new in punc_to_replace:
                                text = text.replace(old, new)
                            text = text.rstrip(" ")
                            enders = {".", "!", "?", "-", ","}
                            if not any(text.endswith(p) for p in enders):
                                text += "."
                            return text

                        sample_text_norm = punc_norm(text_for_sample)
                        sample_text_ids = text_tokenizer.text_to_tokens(sample_text_norm).to(current_device)
                        sot, eot = t3_model.hp.start_text_token, t3_model.hp.stop_text_token
                        sample_text_ids = torch.nn.functional.pad(sample_text_ids, (1, 0), value=sot)
                        sample_text_ids = torch.nn.functional.pad(sample_text_ids, (0, 1), value=eot)
                        repeated_sample_text = torch.cat([sample_text_ids, sample_text_ids], dim=0).unsqueeze(0)

                        # Build T3Cond for sample
                        s3_prompt_sample_seq = s3_prompt_sample.squeeze(0)
                        if s3_prompt_sample_seq.size(0) > t3_cond_prompt_len:
                            cond_tok_sample = s3_prompt_sample_seq[:t3_cond_prompt_len]
                        else:
                            pad_size_s = t3_cond_prompt_len - s3_prompt_sample_seq.size(0)
                            cond_tok_sample = torch.cat([s3_prompt_sample_seq, torch.full((pad_size_s,), SPEECH_VOCAB_SIZE + 1,
                                                                                           device=current_device, dtype=torch.long)], dim=0)

                        t3_cond_sample = T3Cond(
                            speaker_emb=ve_emb_sample,
                            cond_prompt_speech_tokens=cond_tok_sample.unsqueeze(0),
                            emotion_adv=0.5 * torch.ones(1, 1, 1, device=current_device)
                        )
                        # CFG
                        if hasattr(t3_cond_sample, 'repeat_batch') and callable(t3_cond_sample.repeat_batch):
                            t3_cond_sample_b = t3_cond_sample.repeat_batch(2)
                        else:
                            t3_cond_sample_b = T3Cond(
                                speaker_emb=ve_emb_sample.repeat(2, 1, 1),
                                cond_prompt_speech_tokens=cond_tok_sample.unsqueeze(0).repeat(2, 1),
                                emotion_adv=0.5 * torch.ones(2, 1, 1, device=current_device)
                            )

                        with torch.no_grad():
                            gen_s3_b_sample = t3_model.inference(
                                t3_cond=t3_cond_sample_b,
                                text_tokens=repeated_sample_text.squeeze(0),
                                max_new_tokens=1000,
                                temperature=0.7,
                                cfg_weight=0.5
                            )
                        gen_s3_sample = gen_s3_b_sample[0].unsqueeze(0).to(current_device)

                        # Generate mel and audio from text
                        sample_ref_dict = s3mel_model_finetune.embed_ref(ref_wav=sample_wav16, ref_sr=16000,
                                                                         device=current_device)
                        gen_mels_text = s3mel_model_finetune.forward(
                            speech_tokens=gen_s3_sample,
                            ref_wav=None,
                            ref_sr=None,
                            ref_dict=sample_ref_dict,
                            finalize=True
                        )
                        gen_audio_text, _ = vocoder_for_sampling.inference(speech_feat=gen_mels_text)
                        writer.add_audio(f"Sample/Text_Audio_E{epoch + 1}_S{global_step}",
                                         gen_audio_text.squeeze().cpu(), global_step, sample_rate=S3GEN_SR)
                        writer.add_text(f"Sample/Text_E{epoch + 1}_S{global_step}",
                                        sample_text_norm, global_step)

                    except Exception as e:
                        print(f"Error audio sampling step {global_step}: {e}")
                        import traceback; traceback.print_exc()
                s3mel_model_finetune.train()
                t3_model.train()

            global_step += 1

        avg_epoch_loss = epoch_loss_sum / num_samples_processed_epoch if num_samples_processed_epoch > 0 else 0.0
        writer.add_scalar("Loss/train_epoch_avg", avg_epoch_loss, epoch + 1)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.4f}")

        # ======== Evaluation ========
        current_lr_ = optimizer.param_groups[0]['lr']
        avg_val_loss_ = None
        if eval_loader and (epoch + 1) % args.eval_interval == 0:
            s3mel_model_finetune.eval()
            t3_model.eval()
            val_loss_sum_ = 0.0
            num_val_samples_processed = 0
            with torch.no_grad():
                for val_i, val_batch_ in enumerate(tqdm(eval_loader, desc="Evaluating")):
                    if val_batch_ is None:
                        continue
                    try:
                        val_actual_batch_size = val_batch_['speech_token'].size(0)
                        # Reconstruction loss
                        loss_dict_val_ = s3mel_model_finetune.flow.forward(val_batch_, device=current_device)
                        loss_recon_val = loss_dict_val_['loss']

                        # Text→mel loss in validation (optional; here we use only recon)
                        val_loss_sum_ += loss_recon_val.item() * val_actual_batch_size
                        num_val_samples_processed += val_actual_batch_size
                    except Exception as e:
                        print(f"[ERROR] evaluation batch {val_i}: {e}")
                        continue
            avg_val_loss_ = val_loss_sum_ / num_val_samples_processed if num_val_samples_processed > 0 else 0.0
            writer.add_scalar("Loss/val_epoch_avg", avg_val_loss_, epoch + 1)
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss_:.4f}")

            if scheduler is not None and not (args.num_warmup_steps > 0 and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)):
                scheduler.step(avg_val_loss_)
                if optimizer.param_groups[0]['lr'] < current_lr_:
                    print(f"Learning rate reduced by ReduceLROnPlateau to {optimizer.param_groups[0]['lr']}")
            s3mel_model_finetune.train()
            t3_model.train()
        elif not eval_loader and (epoch + 1) % args.eval_interval == 0:
            if scheduler is not None and not (args.num_warmup_steps > 0 and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)):
                scheduler.step(avg_epoch_loss)
                if optimizer.param_groups[0]['lr'] < current_lr_:
                    print(f"Learning rate reduced by ReduceLROnPlateau to {optimizer.param_groups[0]['lr']}")

        # ======== Save checkpoint (includes T3) ========
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path_save = os.path.join(args.output_dir, f"s3mel_t3_epoch{epoch + 1}_step{global_step}.pt")
            save_dict_ = {
                'epoch': epoch + 1,
                'model_state_dict': s3mel_model_finetune.state_dict(),
                't3_state_dict': t3_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': avg_epoch_loss,
                'global_step': global_step,
                'args': vars(args)
            }
            if avg_val_loss_ is not None:
                save_dict_['val_loss'] = avg_val_loss_
            torch.save(save_dict_, ckpt_path_save)
            print(f"Epoch {epoch + 1} Checkpoint saved to {ckpt_path_save}")

    writer.close()
    print("Training finished.")
