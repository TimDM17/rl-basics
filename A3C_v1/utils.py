from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def push_and_pull(opt, lnet, gnet, done, s_image_normalized_for_pull, s_raw_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.
    else:
        s_batch = {"image": v_wrap(s_image_normalized_for_pull[None, :])} 
        _, v_s_ = lnet.forward(s_batch)
        v_s_ = v_s_.data.numpy()[0]

    buffer_v_target = []
    for r in br[::-1]: 
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    normalized_bs_images = np.stack([s["image"] / 255.0 for s in bs])
    batch_obs = {"image": v_wrap(normalized_bs_images)}
    
    loss = lnet.loss_func(
        batch_obs,
        v_wrap(np.array(ba), dtype=np.int64),
        v_wrap(np.array(buffer_v_target)[:, None])
    )

    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

