'''Train CIFAR100 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import wandb


from scipy.optimize import nnls
from models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--optimizer',default="sgd", type=str, help='[momentum,sgd,adam,rmsprop,adagrad,adamw,amsgrad]')
parser.add_argument('--use_wandb', action="store_true", help='Set flag if using wandb.')
parser.add_argument('--sampling', default="normal", type=str, help='[normal,replacement,independent,tau-nice]')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# ==========================
# Data
# ==========================
print('==> Preparing data..')
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# ==========================
# Model
# ==========================
print('==> Building model..')
model = ResNet18()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
if args.optimizer == "momentum":
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
elif args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
elif args.optimizer == "adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
elif args.optimizer == "amsgrad":
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)

tau = 64
q = np.array([1 / args.batch_size] * args.batch_size)
p = np.array([tau / args.batch_size] * args.batch_size)

# ==========================
# Utility: Gradient norm calculation
# ==========================
def get_minibatch_grad_norm(model):
    grad_list = []
    for p in model.parameters():
        if p.grad is not None:
            grad_list.append(p.grad.detach().view(-1))
    if len(grad_list) == 0:
        return 0.0
    grad_vec = torch.cat(grad_list)
    return torch.norm(grad_vec, p=2).item()

# ----- 勾配ベクトルを取る -----
def get_grad_vector(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if len(grads) == 0:
        return torch.tensor([0.0], device=device)
    return torch.cat(grads)

# ----- スナップショット用バッファ -----
snapshots = []
best_train_loss = float("inf")

def compare_gradients(epoch):
    global best_train_loss
    
    # ---- フルバッチ勾配 ----
    g_full_list = get_full_grad_list(model, trainloader, optimizer, args.batch_size, device)
    g_full = torch.tensor(g_full_list, device=device)
    grad_full_norm2 = torch.norm(g_full).item()**2
    
    # ---- ミニバッチ勾配との差 (LHS) ----
    batch_norm_diffs = []
    num_samples = 5
    for _ in range(num_samples):
        inputs, targets = next(iter(trainloader))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        g_batch = get_grad_vector(model)
        diff2 = torch.norm(g_batch - g_full).item()**2
        batch_norm_diffs.append(diff2)
    lhs = np.mean(batch_norm_diffs)  # E[||g-∇f||^2]
    
    # ---- f(x), f* ----
    current_loss = loss.item()
    if current_loss < best_train_loss:
        best_train_loss = current_loss
    f_gap = current_loss - best_train_loss
    
    # ---- スナップショット保存 ----
    snapshots.append((f_gap, grad_full_norm2, lhs))
    if len(snapshots) > 50:  # バッファサイズ制限
        snapshots.pop(0)
    
    # ---- NNLS で A,B,C 推定 ----
    z_hat = {'A': None, 'B': None, 'C': None}
    rhs, residual = None, None
    if len(snapshots) >= 5:
        W = np.stack([
            np.array([s[0] for s in snapshots]),  # f_gap
            np.array([s[1] for s in snapshots]),  # ||grad||^2
            np.ones(len(snapshots))
        ], axis=1)
        b = np.array([s[2] for s in snapshots])  # LHS
        z, _ = nnls(W, b)
        z_hat = {'A': float(z[0]), 'B': float(z[1]), 'C': float(z[2])}
        rhs = float(W[-1] @ z)
        residual = float(b[-1] - rhs)
    
    # ---- WandB ログ ----
    if args.use_wandb:
        log_data = {
            'epoch': epoch,
            'ES_LHS': lhs,
            'grad_full_norm2': grad_full_norm2,
            'f_gap': f_gap
        }
        if rhs is not None:
            log_data.update({
                'ES_RHS': rhs,
                'A': z_hat['A'],
                'B': z_hat['B'],
                'C': z_hat['C'],
                'residual': residual
            })
        wandb.log(log_data)

# ==========================
# Training
# ==========================
def train():
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    grad_norms = []  # ★ 各ミニバッチの勾配ノルムを記録

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if args.sampling == "normal":
            v_np = np.ones(args.batch_size)
        elif args.sampling == "replacement":
            v_np = sampling_replacement(args.batch_size, 32, q)
        elif args.sampling == "independent":
            v_np = sampling_independent(args.batch_size, p)
        elif args.sampling == "tau-nice":
            v_np = sampling_tau_nice(args.batch_size, tau)
        else:
            raise ValueError("Unknown sampling mode")

        v = torch.tensor(v_np, dtype=torch.float32, device=device)
        weighted_loss = (loss * v).mean()

        weighted_loss.backward()

        # ★ ミニバッチ勾配ノルムを記録
        grad_norms.append(get_minibatch_grad_norm(model))

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    training_acc = 100. * correct / total
    norm_result = get_full_grad_list(model, trainloader, optimizer, args.batch_size, device)
    avg_grad_norm = float(np.mean(grad_norms))  # ★ epoch 平均の勾配ノルム

    if args.use_wandb:
        wandb.log({
            'epoch': epoch,
            'training_acc': training_acc,
            'training_loss': train_loss / (batch_idx + 1),
            'norm_result': norm_result,
            'avg_grad_norm': avg_grad_norm  # ★ 追加ログ
        })

# ==========================
# Test
# ==========================
def test():
    global best_acc
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = total_loss / (batch_idx + 1)
    test_acc = 100. * correct / total
    if args.use_wandb:
        wandb.log({'epoch': epoch, 'test_loss': test_loss, 'test_acc': test_acc})

# ==========================
# Main Loop
# ==========================
if args.use_wandb:
    wandb_project_name = "SGD_ES"
    wandb_exp_name = f"{args.optimizer}_{args.sampling}"
    wandb.init(config=args, project=wandb_project_name, name=wandb_exp_name)

print(optimizer)

for epoch in range(200):
    train()
    test()
    compare_gradients(epoch)