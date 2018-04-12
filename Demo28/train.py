import os
import time
import glob
import torch
from torch import optim
from torch import nn
from torchtext import data
from torchtext import datasets
from code.Demo28.model import SNLIClassifier
from code.Demo28.util import get_args

args = get_args()  # 配置参数获取
# 设置gpu服务器块号
if args.gpu != -1:
    torch.cuda.set_device(args.gpu)
#
if args.spinn:
    inputs = datasets.snli.ParsedTextField(lower=args.lower)
    transitions = datasets.snli.ShiftReduceField()
else:
    inputs = data.Field(lower=args.lower)
    transitions = None
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
        os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

# 我的任务中的训练数据是篇章edus 和 edus的每一步对应操作,用才做控制计算新的数据？

# 数据迭代器构建
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, device=args.gpu)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
if config.birnn:
    config.n_cells *= 2

if config.spinn:
    config.lr = 2e-3  # 3e-4
    config.lr_decay_by = 0.75
    config.lr_decay_every = 1  # 0.6
    config.regularization = 0  # 3e-6
    config.mlp_dropout = 0.07
    config.embed_dropout = 0.08  # 0.17
    config.n_mlp_layers = 2
    config.d_tracker = 64
    config.d_mlp = 1024
    config.d_hidden = 300
    config.d_embed = 300
    config.d_proj = 600
    torch.backends.cudnn.enabled = False
else:
    config.regularization = 0

# 模型设计
model = SNLIClassifier(config)

if config.spinn:
    model.out[len(model.out._modules) - 1].weight.data.uniform_(-0.005, 0.005)
if args.word_vectors:
    model.embed.weight.data = inputs.vocab.vectors
if args.gpu != -1:
    model.cuda()
if args.resume_snapshot:
    model.load_state_dict(torch.load(args.resume_snapshot))

# 定义损失函数
loss_function = nn.CrossEntropyLoss()
# opt = optim.Adam(model.parameters(), lr=args.lr)

# 定义优化器
opt = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.9, eps=1e-6, weight_decay=config.regularization)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
print(header)

# 开始训练
for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct = n_total = train_loss = 0
    for batch_idx, batch_data in enumerate(train_iter):
        model.train()  # 之前是y_pred = model(x) 现在直接train，再过城中计算loss的forward求出参数的梯度
        opt.zero_grad()  # 更新参数
        # 权重衰减
        for pg in opt.param_groups:
            pg['lr'] = args.lr * (args.lr_decay_by ** (
                iterations / len(train_iter) / args.lr_decay_every))
        iterations += 1

        # 传入1批数据，返回预测标签，计算正确率
        pred_label = model(batch_data)
        # print(nn.functional.softmax(answer[0]).data.tolist(), batch.label.data[0])
        n_correct += (torch.max(pred_label, 1)[1].view(batch_data.label.size()).data == batch_data.label.data).sum()
        n_total += batch_data.batch_size
        train_acc = 100. * n_correct / n_total

        # 计算loss和学习过程
        loss = loss_function(pred_label, batch_data.label)
        loss.backward()  # 反向传播
        opt.step()  # 更新参数


        # 打印结果

        # train_loss += loss.data[0] * batch.batch_size
        # if iterations % args.save_every == 0:
        #     snapshot_prefix = os.path.join(args.save_path, 'snapshot')
        #     snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(
        #         train_acc, train_loss / n_total, iterations)
        #
        #     torch.save(model.state_dict(), snapshot_path)
        #     for f in glob.glob(snapshot_prefix + '*'):
        #         if f != snapshot_path:
        #             os.remove(f)


        # 评测部分

        # if iterations % args.dev_every == 0:
        #     model.eval()
        #     dev_iter.init_epoch()
        #     n_dev_correct = dev_loss = 0
        #     for dev_batch_idx, dev_batch in enumerate(dev_iter):
        #         pred_label = model(dev_batch)
        #         n_dev_correct += (
        #             torch.max(pred_label, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
        #         dev_loss += loss_function(pred_label, dev_batch.label).data[0] * dev_batch.batch_size
        #     dev_acc = 100. * n_dev_correct / len(dev)
        #     print(dev_log_template.format(time.time() - start,
        #                                   epoch, iterations, 1 + batch_idx, len(train_iter),
        #                                   100. * (1 + batch_idx) / len(train_iter), train_loss / n_total,
        #                                   dev_loss / len(dev), train_acc, dev_acc))
        #     n_correct = n_total = train_loss = 0
        #     if dev_acc > best_dev_acc:
        #         best_dev_acc = dev_acc
        #         snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
        #         snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc,
        #                                                                                            dev_loss / len(dev),
        #                                                                                            iterations)
        #         torch.save(model.state_dict(), snapshot_path)
        #         for f in glob.glob(snapshot_prefix + '*'):
        #             if f != snapshot_path:
        #                 os.remove(f)
        # elif iterations % args.log_every == 0:
        #     print(log_template.format(time.time() - start,
        #                               epoch, iterations, 1 + batch_idx, len(train_iter),
        #                               100. * (1 + batch_idx) / len(train_iter), train_loss / n_total, ' ' * 8,
        #                               n_correct / n_total * 100, ' ' * 12))
        #     n_correct = n_total = train_loss = 0
