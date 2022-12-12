import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
import mindspore.ops.function as F

from sklearn.linear_model import LogisticRegression


_SQRT2=2**0.5
EPS = 1e-15


def diversity_loss(distribution):
    distribution = distribution.squeeze(axis=0).T

    num_cluster, num_instance = distribution.shape

    total_num = num_cluster * num_cluster
    d_1 = F.broadcast_to(ops.ExpandDims()(distribution, 1), (-1, num_cluster, -1)).reshape(total_num, -1)
    d_2 = F.broadcast_to(ops.ExpandDims()(distribution, 0), (num_cluster, -1, -1)).reshape(total_num, -1)

    metric = nn.RootMeanSquareDistance(symmetric=False, distance_metric="euclidean")
    metric.clear()
    dis = metric.update(d_1.sqrt(), d_2.sqrt(), 0) / _SQRT2
    avg_dis = ops.ReduceSum()(dis) / (total_num - num_cluster)
    loss = max(1 - avg_dis, 0)

    return loss


class BasicConv(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, group=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, group=group, has_bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Cell):
    def construct(self, x):
        return ops.Concat(axis=1)(ops.ExpandDims()(ops.ArgMaxWithValue(axis=1)(x)[0], 1), ops.ArgMaxWithValue(axis=1)(ops.ReduceMean()(x, 1)))


class SpatialGate(nn.Cell):
    def __init__(self, k_num=1, activation='sigmoid'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.activation = activation
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, k_num, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def construct(self, x):
        x_compress = self.compress(x)
        scale = self.spatial(x_compress)

        if self.activation == 'sigmoid':
            scale = ops.Sigmoid()(scale) # default
        elif self.activation == 'softmax':
            b_s, C, H, W = scale.shape
            scale = nn.Softmax()(scale.view(b_s, C, -1)).view(b_s, C, H, W)

        # return x * scale
        return scale


class Conv4Block(nn.Cell):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.layers = nn.SequentialCell(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )

    def construct(self,inp):
        return self.layers(inp)


class BackBone_Conv4(nn.Cell):

    def __init__(self,num_channel=64, args=None):
        super().__init__()

        self.args = args

        self.layer1 = Conv4Block(3,num_channel)
        self.layer2 = Conv4Block(num_channel,num_channel)
        self.layer3 = Conv4Block(num_channel,num_channel)
        self.layer4 = Conv4Block(num_channel,num_channel)
        self.maxpool2d = nn.MaxPool2d(2)

        self.spatialGate4 = SpatialGate()

    def construct(self,x):
        args = self.args

        x = self.layer1(x)
        x = self.maxpool2d(x)

        x = self.layer2(x)
        x = self.maxpool2d(x)

        x = self.layer3(x)
        x = self.maxpool2d(x)

        x_up = self.layer4(x)
        gate4 = self.spatialGate4(x_up)
        x = x_up * gate4

        # TODO 注意调整
        # x = self.maxpool2d(x)
        # x_up = x
        # gate4 = self.maxpool2d(gate4)

        return x, x_up, gate4


class AggregatLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super(AggregatLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.node_trans = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // 8), # 64，8
            nn.ReLU(),
            nn.Dense(in_channels // 8, out_channels), # 8, 64
        )

    def construct(self, x, adj, mask=None):
        B, N, _ = adj.shape

        out = ops.MatMul()(adj, x)
        out = out / ops.clip_by_value(adj.sum(axis=-1, keepdims=True), clip_value_min=ms.Tensor(1, ms.float32))

        out = self.node_trans(out)

        if self.normalize:
            out = ops.L2Normalize(axis=-1)(out)

        if mask is not None:
            out = out * mask.view(B, N, 1)

        return out


class GraphEncoder(nn.Cell):

    def __init__(self, args, instance_dim):
        super(GraphEncoder, self).__init__()
        ms.set_context(device_id=args.gpu)

        self.args = args
        self.L = args.graph_dim
        self.t = args.num_cluster # number of clusters

        # 边的数量占节点的比例
        self.num_adj_parm = args.num_adj_parm # this parameter is used to define min graph adjecment len. num_adj_parm * len(bag). 0 - disable

        self.node_init = nn.SequentialCell(
            nn.Dense(instance_dim, instance_dim // 8), # 64，8
            nn.ReLU(),
            nn.Dense(instance_dim // 8, self.L), # 8, 64
        )

        self.GNNlayer1 = AggregatLayer(self.L, self.L)
        # self.GNNlayer2 = AggregatLayer(self.L, self.L)

        self.cluster_classify = nn.Dense(self.L, self.t)

    def init_adj_matrix(self, nodes):
        if self.args.graph_adge == 'soft':
            return self.convert_bag_to_graph_mine_(nodes)
        elif self.args.graph_adge == 'identity':
            return ops.Eye()(self.num_instance, self.num_instance)

    def get_S_by_kmeans(self, x):
        k = KMeans(n_clusters=self.t, max_iter=10, verbose=False)
        k.fit(x)
        return k.centers

    def construct(self, instances):
        b_s, num_desc, D = instances.shape

        nodes = self.node_init(instances)
        # nodes = self.node_init(instances.view(-1, D)).view(b_s, num_desc, D)
        # nodes = instances

        if self.args.graph_adge == 'naive':
            return ops.ReduceMean(axis=0, keepdims=True)(H), 0
        else:
            N_0, A_0 = nodes, self.init_adj_matrix(nodes)

        # Embedding
        N = self.GNNlayer1(N_0, A_0)
        # N = self.GNNlayer2(N, A_0)

        # Clustering
        # S = ops.Tanh()(self.cluster_classify(N))
        # S = self.cluster_classify(N)
        # S = nn.Softmax(axis=-1)(S)

        centers = kmeans(N.view(b_s * num_desc, -1), self.t)
        # centers = self.get_S_by_kmeans(N.view(b_s * num_desc, -1))
        cluster = ((N.view(b_s * num_desc, -1)[:, None, :] - centers[None, :, :])**2).sum(-1).argmin(1)

        depth, on_value, off_value = self.t, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
        S = ops.OneHot()(cluster.view(b_s, num_desc), depth, on_value, off_value)

        cluster_num = ops.ExpandDims()(S.sum(axis=1), axis=-1)
        out = ops.MatMul()(S.transpose(1, 2), N)
        out = (out + ops.ExpandDims()(centers, axis=0)) / (cluster_num + 1)

        out = out.view(b_s, -1)

        return out, 0

    # GNN methods
    def convert_bag_to_graph_mine_(self, bag):
        b_s, num, C = bag.shape
        left = F.broadcast_to(ops.ExpandDims()(bag, axis=1), (-1, num, -1, -1)).reshape(-1, C)
        right = F.broadcast_to(ops.ExpandDims()(bag, axis=2), (-1, -1, num, -1)).reshape(-1, C)

        metric = nn.RootMeanSquareDistance(symmetric=False, distance_metric="euclidean")
        metric.clear()
        distance = metric.update(left, right, 0)
        distance = distance.view(b_s, -1)

        sorted_dists, incide = ops.Sort(axis=-1, descending=False)(distance)

        NUM_ = num * num
        index = int(self.num_adj_parm * NUM_)
        n = ops.ExpandDims()(sorted_dists[:,index], axis=1)

        A = ops.Cast()(distance < n, ms.float32)
        A = A.view(b_s, num, num)

        return A

    def euclidean_distance_(self, X, Y):
        # return pairwise_distance(X, Y, p=2)
        return ops.Sqrt()(ops.tensor_dot(X, X, 1) - 2 * ops.tensor_dot(X, Y, 1) + ops.tensor_dot(Y, Y, 1))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, has_bias=False)


class DropBlock(nn.Cell):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def construct(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.shape[0] * block_mask.shape[1] * block_mask.shape[2] * block_mask.shape[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = ops.Stack()(
            [
                F.broadcast_to(np.arange(self.block_size).view(-1, 1), (self.block_size, self.block_size)).reshape(-1),
                # - left_padding,
                np.tile(np.arange(self.block_size), self.block_size),  # - left_padding
            ]
        ).T
        offsets = ops.Concat(1)(ops.Zeros()((self.block_size ** 2, 2)), offsets)

        if nr_blocks > 0:
            non_zero_idxs = np.tile(non_zero_idxs, (self.block_size ** 2, 1))
            offsets = np.tile(offsets, (nr_blocks, 1)).view(-1, 4)
            offsets = ops.Cast()(offsets, ms.float64)

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = np.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = np.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False, use_maxpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        self.use_maxpool = use_maxpool
        if self.use_se:
            self.se = SELayer(planes, 4)

    def construct(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        if self.use_maxpool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.shape[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = nn.Dropout(keep_prob=self.drop_rate)(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False, args=None):
        super(ResNet, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320, stride=2, drop_rate=drop_rate,
                                       drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640, stride=2, drop_rate=drop_rate,
                                       drop_block=True, block_size=dropblock_size,use_maxpool=False)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = ops.AdaptiveAvgPool2D(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(keep_prob=1 - self.keep_prob)
        self.drop_rate = drop_rate

        self.spatialGate = SpatialGate()

        # self.num_classes = num_classes
        # if self.num_classes > 0:
        #     self.fc = nn.Dense(640, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,use_maxpool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se,use_maxpool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se,use_maxpool)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se,use_maxpool=use_maxpool)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se,use_maxpool=use_maxpool)
            layers.append(layer)

        return nn.SequentialCell(*layers)

    def construct(self, x, is_feat=False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x

        x = self.layer4(x)
        f3 = x

        gate = self.spatialGate(f3)
        x = f3 * gate
        # final_x = gate
        # if self.keep_avg_pool:
        #     x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # feat = x
        # if self.num_classes > 0:
        #     x = self.fc(x)

        # if is_feat:
        #     return [f0, f1, f2, f3, feat], x
        # else:
        return x, f3, gate


def BackBone_ResNet12(keep_prob=1.0, avg_pool=False, drop_rate=0.1, dropblock_size=5, args=None, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, args=None, **kwargs)
    return model


def BackBone_ResNet50(keep_prob=1.0, avg_pool=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def proto_eval_k_shot(all_feature_vector, episode_size, way, shot, query, dim):

    all_feature_vector = all_feature_vector.view(episode_size, way*(shot+query), dim)

    max_index_list = []
    for i in range(episode_size):
        feature_vector = all_feature_vector[i]
        support = feature_vector[:way*shot].view(way,shot,dim)
        centroid = ops.ExpandDims()(support.mean(1), 0)
        query_imgs = ops.ExpandDims()(feature_vector[way*shot:], 1)

        neg_l2_distance = ops.Neg()(((centroid-query_imgs)**2).sum(axis=-1)).view(way*query, way)
        _,max_index = neg_l2_distance.max(1)

        max_index_list.append(max_index)

    return ops.Concat()(max_index_list)


def proto_forward_log_pred(all_feature_vector,dim,args):
    episode_size, shot_num, query_num, way = args.episode_size, args.shot, args.query, args.way

    # b_s = all_feature_vector.shape[0]
    all_feature_vector = all_feature_vector.view(args.episode_size, way*(shot_num+query_num), dim)

    log_prediction = []
    for i in range(episode_size):
        feature_vector = all_feature_vector[i]

        support = feature_vector[:way*shot_num].view(way, shot_num, dim)
        centroid = ops.ExpandDims()(support.mean(1), 0)

        query = ops.ExpandDims()(feature_vector[way*shot_num:].view(-1, dim), 1)

        distance = ops.Neg()(((centroid-query)**2).sum(axis=-1)).view(way*query_num,way)

        log_prediction.append(nn.LogSoftmax(axis=1)(distance))

    return ops.Concat()(log_prediction)


def logistic_regression(feature_vector,way,shot,query,dim):
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')

        support = feature_vector[:way*shot].view(way, shot, dim)
        centroid = support.mean(1)
        query = feature_vector[way*shot:].view(-1, dim)

        support_y = ms.Tensor([i for i in range(way)], dtype=ms.float64)

        clf.fit(centroid, support_y)
        return ms.Tensor(clf.predict(query))


def classify(feature_vector,way,shot,query,dim,type='knn'):
    if type == 'knn':
        return proto_eval_k_shot(feature_vector,way,shot,query,dim)
    elif type == 'LR':
        return logistic_regression(feature_vector,way,shot,query,dim)


class Proto_Model(nn.Cell):

    def __init__(self,way=None,args=None):
        super().__init__()
        num_channel = 64

        if args.backbone == 'conv4':
            self.num_channel = 64
            self.feature_extractor = BackBone_Conv4(self.num_channel, args)
        elif args.backbone == 'resnet12':
            self.feature_extractor = BackBone_ResNet12(args=args)
            self.num_channel = 640
        elif args.backbone == 'resnet50':
            self.feature_extractor = BackBone_ResNet50(num_channel, args)

        self.args = args
        self.way = way
        self.dim = 64

    def get_feature_vector(self,inp):
        pass

    def eval_k_shot(self,inp,way,shot,query):
        feature_vector = self.get_feature_vector(inp)

        max_index = proto_eval_k_shot(feature_vector,
            way = way,
            shot = shot,
            query = query,
            dim = self.dim)

        return max_index

    def construct(self,inp):
        feature_vector = self.get_feature_vector(inp)
        log_prediction = proto_forward_log_pred(feature_vector,
            dim = self.dim,
            args = self.args)

        return log_prediction


class DualAttentionNet(Proto_Model):

    def __init__(self,args=None):
        super().__init__(args=args)
        self.args = args
        self.use_hard_stream = True if (args.hard_stream == 'True') else False

        self.show_cam = False

        global_num = 2 if args.soft_pool == 'GAP_GMP' else 1
        global_dim = self.num_channel * (global_num if args.soft_stream == 'True' else 0)

        mil_dim = args.graph_dim * args.num_cluster

        if self.args.graph_adge == 'naive':
            mil_dim = self.num_channel

        if (self.use_hard_stream == False):
            self.dim = global_dim
        else:
            self.dim = mil_dim  + global_dim # graph

        self.global_gate = SpatialGate(1, activation='sigmoid')

        if self.use_hard_stream:
            if args.backbone == 'conv4':
                channel = 64
            elif args.backbone == 'resnet12':
                channel = 640
            self.instanceMapping = GraphEncoder(args, channel)

    def eval_k_shot(self,inp,episode_size,way,shot,query):
        feature_vector = self.get_feature_vector(inp)

        if isinstance(feature_vector, tuple):
            feature_vector, scores = feature_vector[0], feature_vector[1]

        max_index = proto_eval_k_shot(feature_vector,
            episode_size=episode_size,
            way = way,
            shot = shot,
            query = query,
            dim = self.dim)

        return max_index

    def get_instance_feature_attGraph(self, positive_num=15, feature_map=None, mask=None):
        b_s, C, H, W = feature_map.shape

        # key_descriptors = []
        instance_vector = []
        avg_graph_loss = 0
        selected_x = None

        # sorted_mask, _ = ops.Sort(1)(mask.view(b_s, -1))
        # sigma = sorted_mask[:,(-1 * positive_num)]
        mask = mask.view(b_s, -1)
        _, index = ops.TopK()(mask, positive_num)

        choose_mask = ops.ZerosLike()(mask)
        choose_mask = ops.scatter_nd(index, choose_mask, ops.OnesLike()(mask))

        choose_mask = choose_mask.reshape(b_s, 1, H, W)
        key_descriptors = ops.masked_select(feature_map, choose_mask>0).reshape(b_s, positive_num, C)

        instance_vector, graph_loss = self.instanceMapping(key_descriptors)

        return instance_vector, graph_loss, key_descriptors

    def get_feature_vector(self, inp):
        args = self.args

        episode_size, one_episode_count, C,H,W = inp.shape
        inp = inp.view(episode_size*one_episode_count, C, H, W)

        feature_map, feature_map_up, mask = self.feature_extractor(inp)

        b_s = inp.shape[0]

        graph_loss = 0

        if args.soft_stream == 'True':
            avgpool = ops.AvgPool(kernel_size=feature_map.size(-1))

            gap_vector = avgpool(feature_map)
            gap_vector = gap_vector.view((b_s, -1))

            gmp_vector = avgpool(feature_map)
            gmp_vector = gmp_vector.view((b_s, -1))


            if args.soft_pool == 'GAP':
                glb_vector = gap_vector
            elif args.soft_pool == 'GMP':
                glb_vector = gmp_vector
            else:
                glb_vector = ops.Concat(axis=-1)(gap_vector, gmp_vector)

        if self.use_hard_stream == False:
            return glb_vector, mask, 0
        else:
            instance_vector, graph_loss, key_descriptors = self.get_instance_feature_attGraph(args.positive_num, feature_map_up, mask)

            if args.soft_stream == 'True':
                final_feature = ops.Concat(axis=1)(instance_vector, glb_vector)
            else:
                final_feature = instance_vector

            final_feature = final_feature.view((b_s, -1))

            return final_feature, mask, graph_loss

    def forward(self, inp):
        feature_vector, masks, graph_loss = self.get_feature_vector(inp)

        log_prediction = proto_forward_log_pred(
            feature_vector,
            dim = self.dim,
            args = self.args)

        return log_prediction, masks, graph_loss

