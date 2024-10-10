import gc

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import (GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)


def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_nonascii_tokens(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    # print(f"input:{input_ids.shape}") torch.Size([177])
    embed_weights = get_embedding_matrix(model)
    # model.model.embed_tokens.weight
    # print(f"embed_weights:{embed_weights.shape}")
    # torch.Size([32000, 4096])
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    # print(f"one_hot:{one_hot.shape}")
    # torch.Size([20, 32000])
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    # print(f"input_embeds:{input_embeds.shape}")
    # input_embeds:torch.Size([1, 20, 4096])

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    # detach() 方法用于从计算图中分离一个张量。这意味着在反向传播中，不会计算从这个张量向前传播的梯度。
    # model.model.embed_tokens(input_ids)
    # print(f"embeds:{embeds.shape}")
    # embeds:torch.Size([1, 177, 4096])

    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], # detach 
            input_embeds, 
            embeds[:,input_slice.stop:,:] #detach
        ], 
        dim=1)
    # print(f"full_embeds:{full_embeds.shape}")
    # full_embeds:torch.Size([1, 177, 4096])
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    # print(f"grad:{grad.shape}")
    # grad:torch.Size([20, 32000])

    return grad

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    # print(f"top_indices:{top_indices.shape}")
    # top_indices:torch.Size([20, 256])

    control_toks = control_toks.to(grad.device)
    # adv tokens
    # print(f"control_toks:{control_toks.shape}")
    # torch.Size([20])

    original_control_toks = control_toks.repeat(batch_size, 1)
    # print(f"original_control_toks:{original_control_toks.shape}")
    # original_control_toks:torch.Size([512, 20])
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size, # 步长
        device=grad.device
    ).type(torch.int64)
    # print(f"new_token_pos:{new_token_pos}")
    # new_token_pos:tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    #      0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    #      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,
    #      2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    #      2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    #      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,
    #      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    #      4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    #      5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,
    #      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    #      7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
    #      7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    #      8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,
    #      9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
    #      9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    #     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
    #     11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    #     11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    #     12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    #     13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14,
    #     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    #     14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    #     15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16,
    #     16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    #     16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    #     17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18,
    #     18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    #     18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    #     19, 19, 19, 19, 19, 19, 19, 19], device='cuda:0')
    # print(f"new_token_pos:{new_token_pos.shape}")
    # new_token_pos:torch.Size([512])

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    # print(f"new_token_val:{new_token_val.shape}")
    # new_token_val:torch.Size([512, 1])

    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    # print(f"new_control_toks:{new_control_toks.shape}")
    # new_control_toks:torch.Size([512, 20])

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, indices, count = [], [], 0
    for i in range(control_cand.shape[0]): # 512
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
                indices.append(i)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        if cands:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        else:
            cands = ["<INVALID>"] * len(control_cand)
        # cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands, indices, count


def sample_shorter_control(tokenizer, control_cand, batch_size):
    random.seed(42)
    new_control_cand = []
    for i in range(batch_size):
        cand = random.choice(control_cand)
        tokens = tokenizer(cand, add_special_tokens=False).input_ids
        drop_index = random.randint(0, len(tokens) - 1)
        new_tokens = tokens[:drop_index] + tokens[drop_index + 1:]
        new_cand = tokenizer.decode(new_tokens, skip_special_tokens=True)
        new_control_cand.append(new_cand)
    return new_control_cand


def sample_shorter_control_with_attention(tokenizer, control_cand_ids, attentions, batch_size):
    new_control_cand = []
    indices = []
    attentions_combined = []
    for i in range(len(control_cand_ids)):
        attentions_combined.append(attentions[i].numpy())
        indices += [[i, j] for j in range(len(attentions[i]))]
    attentions_combined = np.concatenate(attentions_combined).reshape(-1)
    prob = 1 / attentions_combined
    prob = prob / prob.sum()
    select = np.random.choice(np.arange(attentions_combined.size), size=batch_size, replace=False, p=prob)
    for s in select:
        index = indices[s]
        ids = control_cand_ids[index[0]]
        new_ids = torch.cat((ids[:index[1]], ids[index[1] + 1:]))
        new_cand = tokenizer.decode(new_ids, skip_special_tokens=True)
        new_control_cand.append(new_cand)
    return new_control_cand


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        # print(f"test_ids:{len(test_ids)}") -> 512 list
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    # print(f"locs:{locs.shape}")
    # locs:torch.Size([512, 20])
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device), #[512, 177]
        1,
        locs,
        test_ids
    ).to(device)
    # print(f"ids:{ids.shape}")
    # torch.Size([512, 177])

    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def get_logits_by_input_ids(model, input_ids_batch, batch_size=512):
    max_len = max([len(input_ids) for input_ids in input_ids_batch])
    ids = torch.zeros(len(input_ids_batch), max_len)
    for i, input_ids in enumerate(input_ids_batch):
        ids[i, :len(input_ids)] = input_ids
    ids = ids.type(input_ids_batch[0].dtype)
    attn_mask = (ids != 0).type(ids.dtype)
    return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids


def forward(*, model, input_ids, attention_mask, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logits = []
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits.detach().cpu())

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)


def target_loss(logits, ids, target_slice, tokenizer):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


def target_loss_batch(logits, ids, target_slice_list):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss = torch.zeros(len(target_slice_list))
    for i, target_slice in enumerate(target_slice_list):
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        loss[i] = crit(logits[i,loss_slice,:], ids[i,target_slice]).mean()
    return loss
