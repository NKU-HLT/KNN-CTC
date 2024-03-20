from typing import List

import torch


def basic_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
) -> List[List[int]]:
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device) # here lack fo 'to(device)'
    cache = model.predictor.init_state(1,
                                       method="zero",
                                       device=encoder_out.device)
    new_cache: List[torch.Tensor] = []
    t = 0
    hyps = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]

        joint_out_step = model.joint(encoder_out_step,
                                     pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0

    return [hyps]


def knn_rnnt_basic_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
    knn_args=None,
    knn_wrapper=None,
) -> List[List[int]]:
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device) # here lack fo 'to(device)'
    cache = model.predictor.init_state(1,
                                       method="zero",
                                       device=encoder_out.device)
    new_cache: List[torch.Tensor] = []
    t = 0
    hyps = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        if prev_out_nblk:
            step_outs = model.predictor.forward_step(pred_input_step, padding,
                                                     cache)  # [1, 1, P]
            pred_out_step, new_cache = step_outs[0], step_outs[1]

        joint_out_step = model.joint(encoder_out_step,
                                     pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)        
        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        
        # here we need to insert knn.
        # we concat encoder_out_step and pred_out_step instead of adding them.
        if joint_out_max != model.blank:
            captured_key = torch.cat((encoder_out_step,pred_out_step), -1)
            confidence,captured_value = torch.max(joint_out_probs,-1)
            captured_value = captured_value.squeeze(-1) 
            if knn_args.build_index:
                knn_wrapper.process(captured_key,captured_value,confidence)
                hyps.append(joint_out_max.item())
            elif knn_args.knn:
                knn_rnnt_softmax_outputs = knn_wrapper.process(captured_key,joint_out_probs)
                _, top1_index = torch.max(knn_rnnt_softmax_outputs,dim=-1)
                hyps.append(top1_index.squeeze().item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache
        
        # delete part of origal rnnt code.
        # if joint_out_max != model.blank:
        #     hyps.append(joint_out_max.item())
        #     prev_out_nblk = True
        #     per_frame_noblk = per_frame_noblk + 1
        #     pred_input_step = joint_out_max.reshape(1, 1)
        #     # state_m, state_c =  clstate_out_m, state_out_c
        #     cache = new_cache

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0

    return [hyps]