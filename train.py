import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
import numpy
import argparse
from DualAttentionNet import DualAttentionNet

context.set_context(mode=context.PYNATIVE_MODE)


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='resnet12')
parser.add_argument('--hard_stream', type=str, default='True')
parser.add_argument('--soft_pool', type=str, default='GAP_GMP')
parser.add_argument('--soft_stream', type=str, default='True')

parser.add_argument("--lr",help="initial learning rate",type=float, default=1e-3)
parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=900)
parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=1)
parser.add_argument('--episode_size', help='the mini-batch size of training', type=int, default=1)
parser.add_argument('--test_episode_size', help='one episode is taken as a mini-batch', type=int, default=1)
parser.add_argument("--episode_train", help="total number of episode on train", type=int, default=1e2)
parser.add_argument("--episode_val", help="total number of episode on val", type=int, default=600)
parser.add_argument("--episode_test", help="total number of episode on test", type=int, default=600)

# common params
parser.add_argument("--img_size", help="resize image size", type=int, default=84)

# key instance params
parser.add_argument("--way", help="way when train", type=int, default=5)
parser.add_argument("--way_test", help="way when test", type=int, default=5)
parser.add_argument("--shot", help="shot", type=int, default=1)
parser.add_argument("--query", help="query", type=int, default=15)

# graph params
parser.add_argument('--positive_num', type=int, default=10)
parser.add_argument('--graph_adge', type=str, default='soft', choices=['hard', 'soft', 'identity', 'naive'])
parser.add_argument('--loss_weight', type=float, default=1)
parser.add_argument('--graph_dim', type=int, default=64)
parser.add_argument('--num_cluster', type=int, default=3)
parser.add_argument('--num_adj_parm', type=float, default=0.3)

args = parser.parse_args()


# The following code is only used to verify the validity of the model
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
# train_dataset = Dataset_csv('../DATASET/CUB', mode='train')
# test_dataset = Dataset_csv('../DATASET/CUB', mode='val')
print(train_dataset.get_col_names())

def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 80)
test_dataset = datapipe(test_dataset, 80)

for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    break

for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break


model = DualAttentionNet(args)
print(model)

# Instantiate loss function and optimizer
# loss_fn = nn.CrossEntropyLoss()
loss_fn = ops.NLLLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)


def train(model, dataset, loss_fn, optimizer, epoch):
    # Define forward function
    weight = ms.Tensor(numpy.random.rand(5), ms.float32)
    def forward_fn(data, label):
        logits = model(data)
        loss, wei = loss_fn(logits, label, weight)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        max_index, _ = ops.ArgMaxWithValue(1)(logits)
        acc = 100 * ops.ReduceSum()(ops.Equal()(max_index, label).astype(ms.int32)) / args.query / args.way
        return loss, acc

    size = dataset.get_dataset_size()
    avg_acc = 0.0
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        label = ms.Tensor([i // args.query for i in range(args.query * args.way)], ms.int32).repeat(args.episode_size)

        loss, acc = train_step(data, label)
        avg_acc += acc
        print(avg_acc)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss}  [{current:>3d}/{size:>3d}]")

    avg_acc = avg_acc / args.episode_size
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch}/{args.epoch}: train_acc: {avg_acc:.3f}")


def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, avg_acc = 0, 0
    for data, label in dataset.create_tuple_iterator():
        label = ms.Tensor([i // args.query for i in range(args.query * args.way)], ms.int32).repeat(args.episode_size)
        max_index = model.eval_k_shot(data, args.episode_size, args.way, args.shot, args.query)
        acc = 100 * ops.ReduceSum()(ops.Equal()(max_index, label).astype(ms.int32)) / args.query / args.way
        avg_acc += acc
        total += len(data)

    avg_acc /= total
    return avg_acc


epochs = args.epoch
best_acc = 0
best_epoch = None
for t in range(epochs):
    print()
    train(model, train_dataset, loss_fn, optimizer, t)
    if (t + 1) % 5 == 0:
        acc = test(model, test_dataset, loss_fn)
        print(f"epoch {t}/{args.epoch}: val_acc: {acc:.3f}")

        if acc > best_acc:
            best_epoch = t
            ms.save_checkpoint(model, "best_model.ckpt")
            print("BEST!")

print("training finished!")
print('------------------------')
print(f'the best epoch is {best_epoch}/{args.epoch}')
print(f'the best val acc is {best_acc}')
print('------------------------')


