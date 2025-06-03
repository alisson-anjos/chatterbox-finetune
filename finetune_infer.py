import torch
import torchaudio
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import torch.multiprocessing as mp


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))) 

from chatterbox.models.s3gen.s3gen import S3Token2Mel, S3Token2Wav 
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR as S3_TOKENIZER_SR
from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer, SPEECH_VOCAB_SIZE 
from chatterbox.models.s3tokenizer import drop_invalid_tokens 
from chatterbox.models.tokenizers import EnTokenizer 
from chatterbox.models.t3 import T3 
from chatterbox.models.t3.modules.cond_enc import T3Cond 
from chatterbox.models.voice_encoder import VoiceEncoder 
from chatterbox.models.s3gen.xvector import CAMPPlus

try:
    from chatterbox.tts import Conditionals 
except ImportError:
    print("[WARN] Could not import Conditionals from chatterbox.tts. TTS mode without ref_wav_path might not work if conds.pt is used.")
    @dataclass
    class Conditionals:
        t3: T3Cond
        gen: dict
        def to(self, device): return self
        @classmethod
        def load(cls, fpath, map_location="cpu"):
            raise NotImplementedError("Conditionals class not found and placeholder cannot load.")


REPO_ID = "ResembleAI/chatterbox"

def load_s3mel_finetuned(checkpoint_path, device):
    model = S3Token2Mel() 
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict_key = 'model_state_dict' if 'model_state_dict' in ckpt else 'state_dict'
        
        state_dict_to_load_from_ckpt = None
        if state_dict_key in ckpt :
            state_dict_to_load_from_ckpt = ckpt[state_dict_key]
        elif isinstance(ckpt, OrderedDict) or isinstance(ckpt, dict): # Se o ckpt já for o state_dict
            state_dict_to_load_from_ckpt = ckpt
        else:
            raise KeyError(f"Checkpoint does not contain a recognizable state_dict (expected keys: '{state_dict_key}' or 'state_dict', or the checkpoint to be a state_dict itself).")
        
        print(f"Fine-tuned S3Token2Mel: Loading state_dict with {len(state_dict_to_load_from_ckpt)} keys.")
        
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load_from_ckpt.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys: print(f"WARNING: Missing keys when loading fine-tuned S3Token2Mel: {missing_keys}")
        if unexpected_keys: print(f"WARNING: Unexpected keys when loading fine-tuned S3Token2Mel: {unexpected_keys}")
        
        if not missing_keys and not unexpected_keys:
             print(f"Fine-tuned S3Token2Mel model loaded successfully from {checkpoint_path}")
        else:
             print(f"Fine-tuned S3Token2Mel loaded from {checkpoint_path} with some key mismatches (see warnings).")

    except Exception as e:
        print(f"Error loading fine-tuned S3Token2Mel checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure the checkpoint path is correct and compatible with S3Token2Mel architecture.")
        raise
    model.to(device)
    model.eval()
    return model

def download_if_needed(repo_id, filename, cache_dir=None):
    try:
        print(f"Downloading {filename} from {repo_id}...")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        print(f"Successfully downloaded {filename} to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Could not download {filename} from {repo_id}: {e}")
        return None

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
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text

def main():
    parser = argparse.ArgumentParser(description="Chatterbox Fine-tuned TTS Inference")
    parser.add_argument('--finetuned_s3mel_checkpoint', type=str, required=True,
                        help="Path to your fine-tuned S3Token2Mel checkpoint (.pt file).")
    parser.add_argument('--ref_wav_path', type=str, default=None, 
                        help="Path to the reference WAV file for voice cloning. If None, uses default conditionals from conds.pt.")
    parser.add_argument('--text', type=str, default=None, 
                        help="Text to synthesize (uses T3 model). If None and --reconstruct_ref_audio is not set, an error will occur.")
    parser.add_argument('--output_wav_path', type=str, default="output_tts_inferred.wav", 
                        help="Path to save the generated WAV file.")
    parser.add_argument('--chatterbox_repo_cache', type=str, default=None,
                        help="Optional local cache directory for Hugging Face Hub models from ResembleAI/chatterbox.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--reconstruct_ref_audio', action='store_true',
                        help="If set, ignores --text and attempts to reconstruct the audio from --ref_wav_path using its own S3 tokens.")

    args = parser.parse_args()

    if not args.reconstruct_ref_audio and not args.text:
        parser.error("Either --text must be provided or --reconstruct_ref_audio must be set (which requires --ref_wav_path).")
    if args.reconstruct_ref_audio and not args.ref_wav_path:
        parser.error("--reconstruct_ref_audio requires --ref_wav_path to be set.")
    
    device = torch.device(args.device)

    s3mel_finetuned_model = load_s3mel_finetuned(args.finetuned_s3mel_checkpoint, device)

    print("Loading pre-trained components from ResembleAI/chatterbox (if needed for TTS mode)...")

    s3gen_full_path = download_if_needed(REPO_ID, "s3gen.safetensors", args.chatterbox_repo_cache)
    if not s3gen_full_path: sys.exit(1)
    temp_s3token2wav_for_vocoder = S3Token2Wav()
    s3gen_full_state_dict = load_safetensors(s3gen_full_path, device='cpu')
    processed_s3gen_state_dict = OrderedDict()
    has_module_prefix = any(k.startswith('module.') for k in s3gen_full_state_dict.keys())
    for k, v_s3gen in s3gen_full_state_dict.items():
        name = k[7:] if has_module_prefix and k.startswith('module.') else k
        processed_s3gen_state_dict[name] = v_s3gen
    temp_s3token2wav_for_vocoder.load_state_dict(processed_s3gen_state_dict, strict=False)
    vocoder = temp_s3token2wav_for_vocoder.mel2wav
    vocoder.to(device).eval()
    print("Pre-trained Vocoder loaded.")
    del temp_s3token2wav_for_vocoder


    s3_tokens_for_s3mel: torch.Tensor
    s3mel_ref_dict_for_s3mel: dict

    if args.reconstruct_ref_audio:
        if not args.ref_wav_path: 
             print("Error: --reconstruct_ref_audio requires --ref_wav_path.")
             sys.exit(1)
        print(f"\n--- RECONSTRUCTION MODE for {args.ref_wav_path} ---")
        
        ref_wav_orig, ref_sr_orig = torchaudio.load(args.ref_wav_path) # type: ignore
        ref_wav_orig_device = ref_wav_orig.to(device) 
        if ref_wav_orig_device.ndim > 1:
            ref_wav_orig_device = ref_wav_orig_device.mean(dim=0, keepdim=True)

        ref_wav_16k_for_s3mel_tokenizer = ref_wav_orig_device
        if ref_sr_orig != S3_TOKENIZER_SR:
            resampler_16k = torchaudio.transforms.Resample(ref_sr_orig, S3_TOKENIZER_SR).to(device)
            ref_wav_16k_for_s3mel_tokenizer = resampler_16k(ref_wav_orig_device)

        with torch.no_grad():
            s3mel_finetuned_model.tokenizer.to(device).eval() # Garante que tokenizer está no device
            s3_tokens_for_s3mel, _ = s3mel_finetuned_model.tokenizer(ref_wav_16k_for_s3mel_tokenizer)
            s3mel_ref_dict_for_s3mel = s3mel_finetuned_model.embed_ref(
                ref_wav=ref_wav_orig_device, 
                ref_sr=ref_sr_orig,
                device=device 
            )
        print(f"S3 tokens for reconstruction extracted from ref audio, shape: {s3_tokens_for_s3mel.shape}")
        if s3_tokens_for_s3mel.numel() == 0:
             print("Error: No S3 tokens extracted from reference audio for reconstruction.")
             sys.exit(1)
    
    else:
        print(f"\n--- TEXT-TO-SPEECH MODE for text: '{args.text}' ---")
        
        tokenizer_json_path = download_if_needed(REPO_ID, "tokenizer.json", args.chatterbox_repo_cache)
        if not tokenizer_json_path: sys.exit(1)
        text_tokenizer = EnTokenizer(tokenizer_json_path)

        official_voice_encoder = VoiceEncoder() 
        ve_path = download_if_needed(REPO_ID, "ve.safetensors", args.chatterbox_repo_cache)
        if not ve_path: sys.exit(1)
        official_voice_encoder.load_state_dict(load_safetensors(ve_path, device='cpu'))
        official_voice_encoder.to(device).eval()

        t3_model = T3()
        t3_path = download_if_needed(REPO_ID, "t3_cfg.safetensors", args.chatterbox_repo_cache)
        if not t3_path: sys.exit(1)
        t3_state_loaded = load_safetensors(t3_path, device='cpu')
        processed_t3_state_dict = t3_state_loaded
        if "model" in t3_state_loaded and isinstance(t3_state_loaded["model"], list) and len(t3_state_loaded["model"]) > 0 and isinstance(t3_state_loaded["model"][0], dict) : 
            processed_t3_state_dict = t3_state_loaded["model"][0]
        elif "state_dict" in t3_state_loaded and isinstance(t3_state_loaded["state_dict"], dict):
            processed_t3_state_dict = t3_state_loaded["state_dict"]
        final_t3_state_dict = OrderedDict()
        has_t3_module_prefix = any(k.startswith('module.') for k in processed_t3_state_dict.keys())
        for k, v_t3 in processed_t3_state_dict.items():
            name_t3 = k[7:] if has_t3_module_prefix and k.startswith('module.') else k
            final_t3_state_dict[name_t3] = v_t3
        t3_model.load_state_dict(final_t3_state_dict)
        t3_model.to(device).eval()

        t3_conditions_tts: T3Cond
        if args.ref_wav_path: # Clonagem de voz para TTS
            print(f"Preparing T3 conditionals from reference audio: {args.ref_wav_path}")
            ref_wav_orig, ref_sr_orig = torchaudio.load(args.ref_wav_path) # type: ignore
            ref_wav_orig_device = ref_wav_orig.to(device) 
            if ref_wav_orig_device.ndim > 1: ref_wav_orig_device = ref_wav_orig_device.mean(dim=0, keepdim=True)
            
            ref_wav_16k_for_cond = ref_wav_orig_device
            if ref_sr_orig != S3_TOKENIZER_SR:
                ref_wav_16k_for_cond = torchaudio.transforms.Resample(ref_sr_orig, S3_TOKENIZER_SR).to(device)(ref_wav_orig_device)

            with torch.no_grad():
                ref_wav_16k_numpy = ref_wav_16k_for_cond.squeeze(0).cpu().numpy()
                list_of_embeddings = official_voice_encoder.embeds_from_wavs([ref_wav_16k_numpy], sample_rate=S3_TOKENIZER_SR)
                official_ve_embedding_numpy = list_of_embeddings[0] 
                official_ve_embedding = torch.from_numpy(official_ve_embedding_numpy).unsqueeze(0).to(device)

                s3mel_finetuned_model.tokenizer.to(device).eval()
                prompt_s3_tokens_for_t3, _ = s3mel_finetuned_model.tokenizer(ref_wav_16k_for_cond) 
                prompt_s3_tokens_for_t3 = prompt_s3_tokens_for_t3.to(device)
            
            t3_cond_prompt_len = t3_model.hp.speech_cond_prompt_len
            conditioned_speech_tokens = prompt_s3_tokens_for_t3
            if prompt_s3_tokens_for_t3.size(1) > t3_cond_prompt_len: 
                conditioned_speech_tokens = prompt_s3_tokens_for_t3[:, :t3_cond_prompt_len]
            elif prompt_s3_tokens_for_t3.size(1) < t3_cond_prompt_len: 
                conditioned_speech_tokens = torch.nn.functional.pad(prompt_s3_tokens_for_t3, (0, t3_cond_prompt_len - prompt_s3_tokens_for_t3.size(1)), value=SPEECH_VOCAB_SIZE + 1)
            
            t3_conditions_obj_tts = T3Cond(speaker_emb=official_ve_embedding, cond_prompt_speech_tokens=conditioned_speech_tokens, emotion_adv=0.5 * torch.ones(1, 1, 1, device=device))
            t3_conditions_tts = t3_conditions_obj_tts.to(device=device)
            print("T3 conditionals prepared from reference audio.")

            with torch.no_grad():
                s3mel_ref_dict_for_s3mel = s3mel_finetuned_model.embed_ref(ref_wav=ref_wav_orig_device, ref_sr=ref_sr_orig, device=device)
            print("S3Token2Mel reference dictionary (ref_dict) prepared from reference audio.")
        else: 
            print("No reference audio for TTS mode, using default built-in conditionals from conds.pt...")
            conds_pt_path = download_if_needed(REPO_ID, "conds.pt", args.chatterbox_repo_cache)
            if not conds_pt_path:
                print("ERROR: Default conds.pt not found and no reference audio provided. Cannot proceed for default TTS mode.")
                sys.exit(1)
            default_conditionals = Conditionals.load(conds_pt_path, map_location=device)
            default_conditionals = default_conditionals.to(device)
            t3_conditions_tts = default_conditionals.t3
            s3mel_ref_dict_for_s3mel = default_conditionals.gen
            print("Using default T3 conditionals and S3Token2Mel ref_dict.")

        text_normalized = punc_norm(args.text or "")
        text_input_ids = text_tokenizer.text_to_tokens(text_normalized).to(device)
        sot = t3_model.hp.start_text_token; eot = t3_model.hp.stop_text_token
        text_input_ids = torch.nn.functional.pad(text_input_ids, (1, 0), value=sot)
        text_input_ids = torch.nn.functional.pad(text_input_ids, (0, 1), value=eot)
        
        print("Generating S3 tokens from text using T3 model...")
        with torch.no_grad():
            text_input_ids_cfg = torch.cat([text_input_ids, text_input_ids], dim=0)
            t3_conds_cfg: T3Cond 
            if hasattr(t3_conditions_tts, 'repeat_batch') and callable(t3_conditions_tts.repeat_batch):
                t3_conds_cfg_temp_tts = t3_conditions_tts.repeat_batch(2); t3_conds_cfg = t3_conds_cfg_temp_tts.to(device=device) 
            else: 
                t3_conds_cfg_obj_batched_tts = T3Cond( 
                    speaker_emb=t3_conditions_tts.speaker_emb.repeat(2,1,1) if t3_conditions_tts.speaker_emb is not None else None,
                    clap_emb=getattr(t3_conditions_tts, 'clap_emb', None).repeat(2,1,1) if hasattr(t3_conditions_tts, 'clap_emb') and t3_conditions_tts.clap_emb is not None else None,
                    cond_prompt_speech_tokens=t3_conditions_tts.cond_prompt_speech_tokens.repeat(2,1) if t3_conditions_tts.cond_prompt_speech_tokens is not None else None,
                    cond_prompt_speech_emb=getattr(t3_conditions_tts,'cond_prompt_speech_emb', None).repeat(2,1,1) if hasattr(t3_conditions_tts, 'cond_prompt_speech_emb') and t3_conditions_tts.cond_prompt_speech_emb is not None else None,
                    emotion_adv=t3_conditions_tts.emotion_adv.repeat(2,1,1) if t3_conditions_tts.emotion_adv is not None else None,
                ); t3_conds_cfg = t3_conds_cfg_obj_batched_tts.to(device=device) 
            
            generated_s3_tokens_batched = t3_model.inference(t3_cond=t3_conds_cfg, text_tokens=text_input_ids_cfg, max_new_tokens=1000, temperature=0.7, cfg_weight=0.5)
            generated_s3_tokens = generated_s3_tokens_batched[0]; generated_s3_tokens = drop_invalid_tokens(generated_s3_tokens) 
            s3_tokens_for_s3mel = generated_s3_tokens.unsqueeze(0).to(device) 
        print(f"S3 tokens generated, shape: {s3_tokens_for_s3mel.shape}")
        if s3_tokens_for_s3mel.numel() == 0: print("Error: T3 model generated empty S3 tokens."); return

    print("Generating mel-spectrogram using fine-tuned S3Token2Mel...")
    with torch.no_grad():
        generated_mel = s3mel_finetuned_model.forward(
            speech_tokens=s3_tokens_for_s3mel, 
            ref_wav=None, ref_sr=None,  
            ref_dict=s3mel_ref_dict_for_s3mel, 
            finalize=True
        )
    print(f"Mel-spectrogram generated, shape: {generated_mel.shape}")

    print("Generating audio from mel-spectrogram using pre-trained HiFT-GAN...")
    with torch.no_grad():
        output_audio, _ = vocoder.inference(speech_feat=generated_mel)
    output_audio_cpu = output_audio.squeeze().cpu()
    print(f"Audio generated, shape: {output_audio_cpu.shape}, dtype: {output_audio_cpu.dtype}")
    print(f"Output audio stats: min={output_audio_cpu.min().item():.2f}, max={output_audio_cpu.max().item():.2f}, mean={output_audio_cpu.mean().item():.3f}")

    output_audio_to_save = output_audio_cpu
    if output_audio_cpu.ndim == 1:
        output_audio_to_save = output_audio_cpu.unsqueeze(0) 
    if output_audio_to_save.dtype != torch.float32: 
        output_audio_to_save = output_audio_to_save.float()
    max_abs_val = torch.max(torch.abs(output_audio_to_save))
    if max_abs_val > 1.0:
        print(f"Audio had max abs value {max_abs_val:.2f}, normalizing...")
        output_audio_to_save = output_audio_to_save / max_abs_val
    elif max_abs_val == 0:
        print(f"Audio is all zeros, not saving.")
        return
    
    try:
        torchaudio.save(args.output_wav_path, output_audio_to_save, S3GEN_SR)
        print(f"Generated audio saved to {args.output_wav_path}")
    except Exception as e_save:
        print(f"Error saving audio with torchaudio: {e_save}")
        try:
            import soundfile as sf
            print("Attempting to save with soundfile as a fallback...")
            sf.write(args.output_wav_path, output_audio_to_save.squeeze(0).numpy(), S3GEN_SR) 
            print(f"Fallback save successful to {args.output_wav_path} using soundfile.")
        except Exception as e_sf:
            print(f"Error saving audio with soundfile: {e_sf}")


if __name__ == '__main__':
    try:
        if torch.cuda.is_available() and mp.get_start_method(allow_none=True) != 'spawn': 
            mp.set_start_method('spawn', force=True) 
    except RuntimeError as e: 
        if "context has already been set" not in str(e).lower() and "cannot start a new process before" not in str(e).lower():
            print(f"Warning: Could not set start method to 'spawn': {e}")
    except Exception as e_mp: 
        print(f"Warning: Issue with multiprocessing setup: {e_mp}")

    main()


# python finetune_infer.py   --finetuned_s3mel_checkpoint "/home/alissonerdx/models/chatterbox/s3gen.pt"  --text "Olá mundo, este é um teste da minha voz clonada em português." --output_wav_path "saida_finetuned_original.wav" --device "cuda" --ref_wav_path ref_kakashi.wav