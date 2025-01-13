import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time  # 新增导入time模块

# import torch.backends.cudnn as cudnn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
from model.stirer import STIRER
from model.crnn import CRNN
from torch.utils.data import ConcatDataset
from utils.loss import ImageLoss
from utils.label_converter import get_charset, strLabelConverter, str_filt
from utils.ssim_psnr import calculate_psnr, SSIM
from dataset import TextPairDataset, LRSTR_collect_fn


def eval(model, eval_datasets):
    print("----------------<Evaluation>----------------")
    model.eval()
    global_accs = []
    global_crnn_accs = []
    # for CE decoder
    decode_mapper = {}
    for i, c in enumerate(charset):
        decode_mapper[i + 1] = c
    decode_mapper[0] = ""
    print(decode_mapper)
    # Evaluation
    for eval_dataset in eval_datasets:
        crnn_accs = []
        print("Evaluating")
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            collate_fn=collect_fn_eval,
        )
        metrics_recorder = {}
        for it, batch in enumerate(eval_data_loader):
            img_hr, img_lr, label_tensors, length_tensors, label_strs, _ = batch
            img_hr = img_hr.cuda()
            img_lr = img_lr.cuda()
            label_tensors = label_tensors.cuda()
            length_tensors = length_tensors.cuda()
            with torch.no_grad():
                sr, logit, srs, logits = model(img_lr)
                logits.append(logit)
                srs.append(sr)
            for step in range(len(srs)):
                if not "acc_" + str(step) in metrics_recorder.keys():
                    metrics_recorder["acc_" + str(step)] = []
                    metrics_recorder["psnr_" + str(step)] = []
                    metrics_recorder["ssim_" + str(step)] = []
                BS = img_hr.size(0)
                if args["sr"]:
                    img_sr = srs[step]
                if args["rec"]:
                    if step == len(srs) - 1:
                        pred_strs = []
                        for i in range(logits[step].size(0)):
                            ids = logits[step][i].tolist()
                            pred_str = ""
                            for j in range(len(ids)):
                                if ids[j] == len(charset) + 1:
                                    break
                                pred_str += decode_mapper[ids[j]]
                            pred_strs.append(pred_str)
                    else:
                        logit = logits[step].softmax(-1)
                        pred_strs = label_converter.decode(
                            torch.argmax(logit, 2).view(-1),
                            torch.IntTensor([logit.size(1)] * BS),
                        )
                        if isinstance(pred_strs, str):
                            pred_strs = [pred_strs]  # 确保pred_strs是列表
                for i in range(BS):
                    if args["sr"]:
                        sr = img_sr[i, ...]
                        hr = img_hr[i, ...]
                        psnr = calculate_psnr(sr.unsqueeze(0), hr.unsqueeze(0)).cpu()
                        ssim = calculate_ssim(sr.unsqueeze(0), hr.unsqueeze(0)).cpu()
                        metrics_recorder["psnr_" + str(step)].append(psnr)
                        metrics_recorder["ssim_" + str(step)].append(ssim)
                    if args["rec"]:
                        if i < len(pred_strs):  # 添加索引检查
                            pred_str = pred_strs[i]
                            gt_str = label_strs[i]
                            if pred_str == gt_str:
                                metrics_recorder["acc_" + str(step)].append(True)
                            else:
                                metrics_recorder["acc_" + str(step)].append(False)
                        else:
                            metrics_recorder["acc_" + str(step)].append(False)
            logits_last = crnn(srs[-1]).softmax(-1).transpose(0, 1)
            pred_strs_crnn = label_converter.decode(
                torch.argmax(logits_last, 2).view(-1),
                torch.IntTensor([logits_last.size(1)] * BS),
            )
            if isinstance(pred_strs_crnn, str):
                pred_strs_crnn = [pred_strs_crnn]  # 确保pred_strs_crnn是列表
            for i in range(BS):
                if i < len(pred_strs_crnn):  # 添加索引检查
                    pred_str = pred_strs_crnn[i]
                    pred_str = str_filt(pred_str, "lower")
                    gt_str = label_strs[i]
                    gt_str = str_filt(gt_str, "lower")
                    if pred_str == gt_str:
                        crnn_accs.append(True)
                    else:
                        crnn_accs.append(False)
                else:
                    crnn_accs.append(False)
                global_crnn_accs.append(crnn_accs[-1])

        for k in metrics_recorder.keys():
            if len(metrics_recorder[k]) == 0:
                metrics_recorder[k].append(-1)
        for step in range(len(srs)):
            print(
                "STEP %d Acc %.2f PSNR %.2f SSIM %.2f"
                % (
                    step,
                    100.0
                    * sum(metrics_recorder["acc_" + str(step)])
                    / len(metrics_recorder["acc_" + str(step)]),
                    sum(metrics_recorder["psnr_" + str(step)])
                    / len(metrics_recorder["psnr_" + str(step)]),
                    100.0
                    * sum(metrics_recorder["ssim_" + str(step)])
                    / len(metrics_recorder["ssim_" + str(step)]),
                ),
                flush=True,
            )
        # add the last
        step = len(srs) - 1
        global_accs.append(
            sum(metrics_recorder["acc_" + str(step)])
            / len(metrics_recorder["acc_" + str(step)])
        )
        print("CRNN Acc %.2f" % (100.0 * sum(crnn_accs) / len(crnn_accs)))
    final_acc = sum(global_accs) / len(global_accs)
    print("Avg CRNN Acc %.2f" % (100.0 * sum(global_crnn_accs) / len(global_crnn_accs)))
    model.train()
    return final_acc


args = {
    "exp_name": "STIRER_S",
    "batch_size": 64,  # 128
    "multi_card": False,
    "train_dataset": [
        "dataset/syth80k/train",
    ],
    "eval_dataset": [
        "dataset/syth80k/test",
    ],
    "Epoch": 8,
    "alpha": 0.5,
    "print_iter": 100,
    "num_gpu": 1,
    "eval_iter": 200,
    "resume": "",
    "seed": 3721,
    "sr": True,
    "rec": True,
    "multi_stage": False,
    "charset": 37,
    "ar": True,
}
torch.manual_seed(args["seed"])  # cpu
torch.cuda.manual_seed(args["seed"])  # gpu

np.random.seed(args["seed"])  # numpy
random.seed(args["seed"])  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STIRER(
    upscale=2,
    img_size=(16, 64),
    window_size=(16, 2),
    img_range=1.0,
    depths=[3, 4, 5],
    embed_dim=48,
    num_heads=[6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffledirect",
    is_eval=False,
).to(device)
# model = Coding(args).to(device)
crnn = CRNN(32, 1, 37, 256).cuda()
crnn.load_state_dict(
    torch.load("dataset/mydata/crnn.pth", map_location="cpu"), strict=True
)
crnn.eval()
sr_param_list = []
for n, p in model.named_parameters():
    if "upsample" in n or "conv_after_body" in n:
        sr_param_list.append(id(p))
if args["resume"] != "":
    print("Loading parameters from", args["resume"])
    params = torch.load(args["resume"], map_location="cpu")
    model.load_state_dict(params, strict=False)
if args["multi_stage"]:
    params_sr = torch.load("ckpt/coging_base_sronly.pth", map_location="cpu")
    params_rec = torch.load("ckpt/coging_base_S_reconly_v2.pth", map_location="cpu")
    params = {}
    for k in params_sr.keys():
        params[k] = params_sr[k]
    for k in params_rec.keys():
        params[k] = params_rec[k]
    for q in params_rec.keys():
        # copy the vit parameters
        if "rec_backbone" in q:
            for i in range(args["unshared_encoder_layers"]):
                params[
                    "unshared_block_" + str(i) + "." + ".".join(q.split(".")[1:])
                ] = params_rec[q]
    print("Loading the following params:", params.keys())
    model.load_state_dict(params, strict=False)
    # for name, param in model.named_parameters():
    #     param.requires_grad=False
    #     print(name,type(param))
#     for param in model.parameters():
#         print(param.requires_grad)
# os._exit(233)
if args["multi_card"]:
    model = torch.nn.DataParallel(model, device_ids=range(args["num_gpu"]))
# x = torch.randn(4,4,16,64).to(device)
# model(x)
# os._exit(2333)
train_dataset = ConcatDataset(
    [
        TextPairDataset(root, max_len=20, train=True, args=args)
        for root in args["train_dataset"]
    ]
)
collect_fn = LRSTR_collect_fn(args=args)
collect_fn_eval = LRSTR_collect_fn(train=False, args=args)
eval_datasets = [
    TextPairDataset(root, args=args) for root in args["eval_dataset"]
]
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args["batch_size"],
    shuffle=True,
    num_workers=16,
    drop_last=True,
    collate_fn=collect_fn,
)
loss_pixel = ImageLoss()
loss_ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
# log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_().cuda()
# targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
# input_lengths = torch.full((16,), 50, dtype=torch.long)
# target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
# print(log_probs.shape, targets.shape, input_lengths.shape, target_lengths.shape)
# loss = loss_ctc(log_probs, targets, input_lengths, target_lengths)
# loss.backward()
# print("???")
# os._exit(233)
# optimizer = optim.AdamW(model.parameters(), lr=5e-5)
base_params = filter(lambda p: id(p) not in sr_param_list, model.parameters())
sr_params = filter(lambda p: id(p) in sr_param_list, model.parameters())
optimizer = optim.AdamW(
    [
        {"params": base_params},
        {"params": sr_params, "lr": 5e-4},
    ],
    lr=5e-4,
)
charset = get_charset(args["charset"])
label_converter = strLabelConverter(get_charset(args["charset"]))
calculate_ssim = SSIM()
max_acc = 0.0
start_time = time.time()  # 记录开始时间
scaler = torch.cuda.amp.GradScaler()

for epoch in range(args["Epoch"]):
    for it, batch in enumerate(train_loader):
        img_hr, img_lr, label_tensors, length_tensors, label_strs, label_ce = batch
        assert label_tensors.size(0) == length_tensors.sum(), "length not equal!"
        img_hr = img_hr.to(device)
        img_lr = img_lr.to(device)
        label_tensors = label_tensors.to(device)
        length_tensors = length_tensors.to(device)
        label_ce = label_ce.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            sr, ar_logit, srs, logits = model(img_lr, label_ce)
            srs.append(sr)
            lengths_input = (
                torch.zeros_like(length_tensors).fill_(logits[0].size(1)).long()
            )
            loss_sr, loss_rec = 0.0, (len(logits) + 1) * ar_logit.loss.mean()

            if args["sr"]:
                for step, sr in enumerate(srs):
                    loss_sr += (step + 1) * loss_pixel(sr, img_hr).mean()
            if args["rec"]:
                for step, logit in enumerate(logits):
                    logit = torch.nn.functional.log_softmax(logit, -1)
                    loss_rec += (step + 1) * loss_ctc(
                        log_probs=logit.transpose(0, 1),
                        targets=label_tensors.long(),
                        input_lengths=lengths_input.long(),
                        target_lengths=length_tensors.long(),
                    )
            loss = args["alpha"] * loss_sr + (1 - args["alpha"]) * loss_rec

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if it % args["print_iter"] == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_iter = elapsed_time / (epoch * len(train_loader) + it + 1)
            remaining_iters = (args["Epoch"] - epoch - 1) * len(train_loader) + (
                len(train_loader) - it - 1
            )
            remaining_time = remaining_iters * avg_time_per_iter
            print(
                "EPOCH[%d] ITER[%d]/[%d] loss_tot[%.5f] loss_sr[%.5f] loss_rec[%.5f] Elapsed[%.2f]s Remaining[%.2f]s"
                % (
                    epoch,
                    it,
                    len(train_loader),
                    loss,
                    loss_sr,
                    loss_rec,
                    elapsed_time,
                    remaining_time,
                ),
                flush=True,
            )

        if it % args["eval_iter"] == 1 and it != 1:
            acc = eval(model, eval_datasets)
            if acc > max_acc:
                max_acc = acc
                print("Saving Best")
                if args["multi_card"]:
                    save_dict = model.module.state_dict()
                else:
                    save_dict = model.state_dict()
                torch.save(save_dict, os.path.join("ckpt", args["exp_name"] + ".pth"))

    if epoch in [4]:
        print("Reduce LR")
        lr = optimizer.param_groups[0]["lr"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr / 2
