import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from model.internvl.modeling_internvl_chat import InternVLChatModel
from omegaconf import OmegaConf
import os
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from model.internvl.conversation import get_conv_template
from transformers import AutoTokenizer
from tqdm import tqdm, trange

sample_scheduler = DDIMScheduler(
    beta_schedule          = "scaled_linear",
    beta_start             = 0.00085,
    beta_end               = 0.012,
    num_train_timesteps    = 1000,
    clip_sample            = False,
    prediction_type        = "v_prediction",
    set_alpha_to_one       = True,
    steps_offset           = 1,
    trained_betas          = None,
    timestep_spacing       = "trailing",
    rescale_betas_zero_snr = True
)

def diff_generate(feature, diff_head):
    sample_scheduler.set_timesteps(50)
    B = feature.shape[0]

    pred_latents = torch.randn((B, 16), device=feature.device, dtype=feature.dtype)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in sample_scheduler.timesteps:
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        with torch.no_grad():
            t_sample = torch.as_tensor([t], device=feature.device)
            noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
            pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

    return pred_latents


@torch.no_grad()
def sample():
    from runner.down_proj.joint_training import equip_internvl
    # exp_dir = "/data/phd/jinjiachun/experiment/pos/0908_both_task"
    exp_dir = "/data/phd/jinjiachun/experiment/down_proj/0917_joint_training_llm_off"

    exp_name = exp_dir.split("/")[-1]
    step = 36000
    device = "cuda:0"
    dtype = torch.float16

    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)


    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = equip_internvl(internvl, config.model)

    ckpt_path = os.path.join(exp_dir, f"internvl-down_proj-{step}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing: {m}")
    print(f"unmatched: {u}")

    internvl = internvl.to(device, dtype).eval()

    # ----- sampling -----
    IMG_START_TOKEN = "<img>"
    prompts = [
        "A yellow broccoli.",
        "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        "A soft, natural portrait photograph captures a young woman with fair skin and long, ash-blonde hair cascading gently over her shoulders. At the very bottom of the frame, in simple, elegant lettering, appears the phrase 'BE KIND'",
    ]
    cfg_scale = 3

    for idx, prompt_txt in enumerate(prompts):
        template = get_conv_template("internvl2_5")
        prompt = f"Generate an image: {prompt_txt}"
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt() + IMG_START_TOKEN

        print(prompt)

        tokenizer_output = tokenizer(
            [prompt],
            padding        = True,
            padding_side   = "left",
            truncation     = True,
            return_tensors = "pt",
        )
        input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        text_embedding = internvl.language_model.get_input_embeddings()(input_ids).to(device)
        print(text_embedding.shape)

        generated_tokens = torch.zeros((1, 256, 16)).to(device, dtype)
        
        # 初始化past_key_values
        past_key_values = None
        hidden_states_store = []
        for i in trange(256):
            # 第一次迭代使用完整的text_embedding，后续只使用新生成的token
            if i == 0:
                current_input = text_embedding
            else:
                current_input = img_embeds.unsqueeze(dim=1)

            outputs = internvl.language_model.model(
                inputs_embeds=current_input,
                use_cache=True, 
                past_key_values=past_key_values
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            hidden_states_store.append(hidden_states[0].unsqueeze(0))

            z = hidden_states[:, -1, :]

            next_token = diff_generate(z, internvl.diff_head)
            img_embeds = internvl.new_mlp2(next_token)

            generated_tokens[:, i] = next_token

        print(generated_tokens.shape)


        tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # for i in range(clip_features.shape[0]):
        visual_feature = internvl.new_mlp1(generated_tokens)
        question = '<image>\nPlease describe the image in detail.'
        response = internvl.chat_with_visual_feature(tokenizer, visual_feature, question, generation_config)
        print(f'User: {question}\nAssistant: {response}')

if __name__ == "__main__":
    sample()