import torch

def embed_prompt(pipe, prompt, padding="max_length", device='cpu'):
    text_inputs = pipe.tokenizer(
        prompt,
        padding=padding,
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = pipe.tokenizer.batch_decode(
            untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
        )

    attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds