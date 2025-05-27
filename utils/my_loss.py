import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchsnooper
from torch.autograd import Variable


def cal_distance(x_feat, y_feat):
    return 0


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def EU_dist(feat, proto):
    d_matrix = torch.zeros(feat.shape[0], proto.shape[0]).cuda()
    for i in range(feat.shape[0]):
        for j in range(proto.shape[0]):
            d = torch.sqrt(torch.dot((feat[i] - proto[j]), (feat[i] - proto[j])))
            d_matrix[i, j] = d
    return d_matrix


class SingleSideCELoss(nn.Module):
    def __init__(self):
        super(SingleSideCELoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss()

    # @torchsnooper.snoop()

    def forward(self, features, class_prototypes, domain):
        sim = -EU_dist(features, class_prototypes).cuda()
        return self.loss_function(sim, domain.cuda())


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([0.5, 0.5])
        # if isinstance(alpha, (float, int, long)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        # if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    # @torchsnooper.snoop()
    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared).cuda()

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels.cuda()).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    # got the dot product between all embeddings
    cor_mat = torch.matmul(x, x.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    norm_mat = cor_mat.diag()

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = F.relu(distances)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).cuda().byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels  # Combine the two masks

    return mask


class AIAWLoss(nn.Module):
    def __init__(self, reduction=None):
        super(AIAWLoss, self).__init__()
        self.reduction = reduction

    def forward(self, f_map_real, f_map_fake, eye_real, eye_fake, mask_matrix_real, mask_matrix_fake, margin_real,
                margin_fake, num_remove_cov_real, num_remove_cov_fake):
        f_cov_real, B = get_covariance_matrix(f_map_real, eye_real)
        f_cov_masked_real = f_cov_real * mask_matrix_real

        f_cov_fake, B = get_covariance_matrix(f_map_fake, eye_fake)
        f_cov_masked_fake = f_cov_fake * mask_matrix_fake

        off_diag_sum_real = torch.sum(torch.abs(f_cov_masked_real), dim=(1, 2), keepdim=True) - margin_real  # B X 1 X 1
        loss_real = torch.clamp(torch.div(off_diag_sum_real, num_remove_cov_real), min=0)  # B X 1 X 1
        loss_real = torch.sum(loss_real) / B

        off_diag_sum_fake = torch.sum(torch.abs(f_cov_masked_fake), dim=(1, 2), keepdim=True) - margin_fake  # B X 1 X 1
        loss_fake = torch.clamp(torch.div(off_diag_sum_fake, num_remove_cov_fake), min=0)  # B X 1 X 1
        loss_fake = torch.sum(loss_fake) / B

        loss = (loss_real + loss_fake) / 2

        return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

    return f_cor, B


class AMSoftmaxLoss(nn.Module):
    '''drives from the PatchNet:
    https://arxiv.org/abs/2203.14325
    '''
    def __init__(self, s=30.0, m_l=0.4, m_s=0.1):
        '''
        AM Softmax Loss
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l]

    def forward(self, score, labels):
        '''
        input:
            score shape (N, class)
            labels shape (N)
        '''
        assert len(score) == len(labels)
        assert torch.min(labels) >= 0

        m = torch.tensor([self.m[ele] for ele in labels]).to(score.device)
        numerator = self.s * (torch.diagonal(score.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((score[i, :y], score[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0) # (N.1)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

        L = numerator - torch.log(denominator)

        return - torch.mean(L)

class MeanAngleLoss(nn.Module):
    def __init__(self):
        super(MeanAngleLoss, self).__init__()

    def forward(self, features, label, domain, taul=1.0, taus=0.85):
        '''
        :param features: 3 * [B, D]
        :param label: [B,]
        :param domain: [B,]
        :return:
        '''
        m1, m2, m3 = features[0], features[1], features[2]
        d1idx = domain.cuda() == 0
        if d1idx.sum() > 0:
            d1m1, d1m2, d1m3 = m1[d1idx], m2[d1idx], m3[d1idx]
            d1label = label[d1idx]
            d1idx_real = d1label == 1
            d1idx_fake = d1label == 0

            if d1idx_real.sum() > 0:
                d1m1_real = d1m1[d1idx_real] / torch.norm(d1m1[d1idx_real], p=2, dim=1, keepdim=True)
                d1m1_real = d1m1_real.mean(dim=0) / torch.norm(d1m1_real.mean(dim=0), p=2, dim=0)
                d1m2_real = d1m2[d1idx_real] / torch.norm(d1m2[d1idx_real], p=2, dim=1, keepdim=True)
                d1m2_real = d1m2_real.mean(dim=0) / torch.norm(d1m2_real.mean(dim=0), p=2, dim=0)
                d1m3_real = d1m3[d1idx_real] / torch.norm(d1m3[d1idx_real], p=2, dim=1, keepdim=True)
                d1m3_real = d1m3_real.mean(dim=0) / torch.norm(d1m3_real.mean(dim=0), p=2, dim=0)
            if d1idx_fake.sum() > 0:
                d1m1_fake = d1m1[d1idx_fake] / torch.norm(d1m1[d1idx_fake], p=2, dim=1, keepdim=True)
                d1m1_fake = d1m1_fake.mean(dim=0) / torch.norm(d1m1_fake.mean(dim=0), p=2, dim=0)
                d1m2_fake = d1m2[d1idx_fake] / torch.norm(d1m2[d1idx_fake], p=2, dim=1, keepdim=True)
                d1m2_fake = d1m2_fake.mean(dim=0) / torch.norm(d1m2_fake.mean(dim=0), p=2, dim=0)
                d1m3_fake = d1m3[d1idx_fake] / torch.norm(d1m3[d1idx_fake], p=2, dim=1, keepdim=True)
                d1m3_fake = d1m3_fake.mean(dim=0) / torch.norm(d1m3_fake.mean(dim=0), p=2, dim=0)
            if d1idx_real.sum() > 0:
                d1_theta12_real = torch.dot(d1m1_real.squeeze(), d1m2_real.squeeze())
                d1_theta13_real = torch.dot(d1m1_real.squeeze(), d1m3_real.squeeze())
                d1_theta23_real = torch.dot(d1m2_real.squeeze(), d1m3_real.squeeze())
            if d1idx_fake.sum() > 0:
                d1_theta12_fake = torch.dot(d1m1_fake.squeeze(), d1m2_fake.squeeze())
                d1_theta13_fake = torch.dot(d1m1_fake.squeeze(), d1m3_fake.squeeze())
                d1_theta23_fake = torch.dot(d1m2_fake.squeeze(), d1m3_fake.squeeze())

        d2idx = domain.cuda() == 1
        if d2idx.sum() > 0:
            d2m1, d2m2, d2m3 = m1[d2idx], m2[d2idx], m3[d2idx]
            d2label = label[d2idx]
            d2idx_real = d2label == 1
            d2idx_fake = d2label == 0

            if d2idx_real.sum() > 0:
                d2m1_real = d2m1[d2idx_real] / torch.norm(d2m1[d2idx_real], p=2, dim=1, keepdim=True)
                d2m1_real = d2m1_real.mean(dim=0) / torch.norm(d2m1_real.mean(dim=0), p=2, dim=0)
                d2m2_real = d2m2[d2idx_real] / torch.norm(d2m2[d2idx_real], p=2, dim=1, keepdim=True)
                d2m2_real = d2m2_real.mean(dim=0) / torch.norm(d2m2_real.mean(dim=0), p=2, dim=0)
                d2m3_real = d2m3[d2idx_real] / torch.norm(d2m3[d2idx_real], p=2, dim=1, keepdim=True)
                d2m3_real = d2m3_real.mean(dim=0) / torch.norm(d2m3_real.mean(dim=0), p=2, dim=0)
            if d2idx_fake.sum() > 0:
                d2m1_fake = d2m1[d2idx_fake] / torch.norm(d2m1[d2idx_fake], p=2, dim=1, keepdim=True)
                d2m1_fake = d2m1_fake.mean(dim=0) / torch.norm(d2m1_fake.mean(dim=0), p=2, dim=0)
                d2m2_fake = d2m2[d2idx_fake] / torch.norm(d2m2[d2idx_fake], p=2, dim=1, keepdim=True)
                d2m2_fake = d2m2_fake.mean(dim=0) / torch.norm(d2m2_fake.mean(dim=0), p=2, dim=0)
                d2m3_fake = d2m3[d2idx_fake] / torch.norm(d2m3[d2idx_fake], p=2, dim=1, keepdim=True)
                d2m3_fake = d2m3_fake.mean(dim=0) / torch.norm(d2m3_fake.mean(dim=0), p=2, dim=0)
            if d2idx_real.sum() > 0:
                d2_theta12_real = torch.dot(d2m1_real.squeeze(), d2m2_real.squeeze())
                d2_theta13_real = torch.dot(d2m1_real.squeeze(), d2m3_real.squeeze())
                d2_theta23_real = torch.dot(d2m2_real.squeeze(), d2m3_real.squeeze())
            if d2idx_fake.sum() > 0:
                d2_theta12_fake = torch.dot(d2m1_fake.squeeze(), d2m2_fake.squeeze())
                d2_theta13_fake = torch.dot(d2m1_fake.squeeze(), d2m3_fake.squeeze())
                d2_theta23_fake = torch.dot(d2m2_fake.squeeze(), d2m3_fake.squeeze())

        d3idx = domain.cuda() == 2
        if d3idx.sum() > 0:
            d3m1, d3m2, d3m3 = m1[d3idx], m2[d3idx], m3[d3idx]
            d3label = label[d3idx]
            d3idx_real = d3label == 1
            d3idx_fake = d3label == 0
            if d3idx_real.sum() > 0:
                d3m1_real = d3m1[d3idx_real] / torch.norm(d3m1[d3idx_real], p=2, dim=1, keepdim=True)
                d3m1_real = d3m1_real.mean(dim=0) / torch.norm(d3m1_real.mean(dim=0), p=2, dim=0)
                d3m2_real = d3m2[d3idx_real] / torch.norm(d3m2[d3idx_real], p=2, dim=1, keepdim=True)
                d3m2_real = d3m2_real.mean(dim=0) / torch.norm(d3m2_real.mean(dim=0), p=2, dim=0)
                d3m3_real = d3m3[d3idx_real] / torch.norm(d3m3[d3idx_real], p=2, dim=1, keepdim=True)
                d3m3_real = d3m3_real.mean(dim=0) / torch.norm(d3m3_real.mean(dim=0), p=2, dim=0)

            if d3idx_fake.sum() > 0:
                d3m1_fake = d3m1[d3idx_fake] / torch.norm(d3m1[d3idx_fake], p=2, dim=1, keepdim=True)
                d3m1_fake = d3m1_fake.mean(dim=0) / torch.norm(d3m1_fake.mean(dim=0), p=2, dim=0)
                d3m2_fake = d3m2[d3idx_fake] / torch.norm(d3m2[d3idx_fake], p=2, dim=1, keepdim=True)
                d3m2_fake = d3m2_fake.mean(dim=0) / torch.norm(d3m2_fake.mean(dim=0), p=2, dim=0)
                d3m3_fake = d3m3[d3idx_fake] / torch.norm(d3m3[d3idx_fake], p=2, dim=1, keepdim=True)
                d3m3_fake = d3m3_fake.mean(dim=0) / torch.norm(d3m3_fake.mean(dim=0), p=2, dim=0)
            if d3idx_real.sum() > 0:
                d3_theta12_real = torch.dot(d3m1_real.squeeze(), d3m2_real.squeeze())
                d3_theta13_real = torch.dot(d3m1_real.squeeze(), d3m3_real.squeeze())
                d3_theta23_real = torch.dot(d3m2_real.squeeze(), d3m3_real.squeeze())
            if d3idx_fake.sum() > 0:
                d3_theta12_fake = torch.dot(d3m1_fake.squeeze(), d3m2_fake.squeeze())
                d3_theta13_fake = torch.dot(d3m1_fake.squeeze(), d3m3_fake.squeeze())
                d3_theta23_fake = torch.dot(d3m2_fake.squeeze(), d3m3_fake.squeeze())


        loss12, loss13, loss23 = 0, 0, 0
        if d1idx.sum() > 0 and d2idx.sum() > 0 and d3idx.sum() > 0:
            if d1idx_real.sum() > 0 and d2idx_real.sum() > 0 and d3idx_real.sum() > 0:
                d12_theta11_real = torch.dot(d1m1_real, d2m1_real)
                d12_theta22_real = torch.dot(d1m2_real, d2m2_real)
                d12_theta33_real = torch.dot(d1m3_real, d2m3_real)
                d13_theta11_real = torch.dot(d1m1_real, d3m1_real)
                d13_theta22_real = torch.dot(d1m2_real, d3m2_real)
                d13_theta33_real = torch.dot(d1m3_real, d3m3_real)
                d23_theta11_real = torch.dot(d2m1_real, d3m1_real)
                d23_theta22_real = torch.dot(d2m2_real, d3m2_real)
                d23_theta33_real = torch.dot(d2m3_real, d3m3_real)
                loss12 += (d12_theta11_real - taul) ** 2 + (d12_theta22_real - taul) ** 2 + (d12_theta33_real - taul) ** 2
                loss12 += (d1_theta12_real - d2_theta12_real) ** 2 + (d1_theta13_real - d2_theta13_real) ** 2  + (d1_theta23_real - d2_theta23_real) ** 2
                loss13 += (d13_theta11_real - taul) ** 2 + (d13_theta22_real - taul) ** 2 + (d13_theta33_real - taul) ** 2
                loss13 += (d1_theta12_real - d3_theta12_real) ** 2 + (d1_theta13_real - d3_theta13_real) ** 2 + (d1_theta23_real - d3_theta23_real) ** 2
                loss23 += (d23_theta11_real - taul) ** 2 + (d23_theta22_real - taul) ** 2 + (d23_theta33_real - taul) ** 2
                loss23 += (d2_theta12_real - d3_theta12_real) ** 2 + (d2_theta13_real - d3_theta13_real) ** 2 + (d2_theta23_real - d3_theta23_real) ** 2
            if d1idx_fake.sum() > 0 and d2idx_fake.sum() > 0 and d3idx_fake.sum() > 0:
                d12_theta11_fake = torch.dot(d1m1_fake, d2m1_fake)
                d12_theta22_fake = torch.dot(d1m2_fake, d2m2_fake)
                d12_theta33_fake = torch.dot(d1m3_fake, d2m3_fake)
                d13_theta11_fake = torch.dot(d1m1_fake, d3m1_fake)
                d13_theta22_fake = torch.dot(d1m2_fake, d3m2_fake)
                d13_theta33_fake = torch.dot(d1m3_fake, d3m3_fake)
                d23_theta11_fake = torch.dot(d2m1_fake, d3m1_fake)
                d23_theta22_fake = torch.dot(d2m2_fake, d3m2_fake)
                d23_theta33_fake = torch.dot(d2m3_fake, d3m3_fake)
                loss12 += (d12_theta11_fake - taus) ** 2 + (d12_theta22_fake - taus) ** 2 + (d12_theta33_fake - taus) ** 2
                loss12 += (d1_theta12_fake - d2_theta12_fake) ** 2 + (d1_theta13_fake - d2_theta13_fake) ** 2 + (d1_theta23_fake - d2_theta23_fake) ** 2
                loss13 += (d13_theta11_fake - taus) ** 2 + (d13_theta22_fake - taus) ** 2 + (d13_theta33_fake - taus) ** 2
                loss13 += (d1_theta12_fake - d3_theta12_fake) ** 2 + (d1_theta13_fake - d3_theta13_fake) ** 2 + (d1_theta23_fake - d3_theta23_fake) ** 2
                loss23 += (d23_theta11_fake - taus) ** 2 + (d23_theta22_fake - taus) ** 2 + (d23_theta33_fake - taus) ** 2
                loss23 += (d2_theta12_fake - d3_theta12_fake) ** 2 + (d2_theta13_fake - d3_theta13_fake) ** 2 + (d2_theta23_fake - d3_theta23_fake) ** 2
                #print(d3_theta12_fake, d3_theta13_fake)
            #print(loss12, loss13, loss23)
            return (loss12 + loss13 + loss23) / 3
        elif d1idx.sum() > 0 and d2idx.sum() > 0:
            if d1idx_real.sum() > 0 and d2idx_real.sum() > 0:
                d12_theta11_real = torch.dot(d1m1_real, d2m1_real)
                d12_theta22_real = torch.dot(d1m2_real, d2m2_real)
                d12_theta33_real = torch.dot(d1m3_real, d2m3_real)
                loss12 += (d12_theta11_real - taul) ** 2 + (d12_theta22_real - taul) ** 2 + (d12_theta33_real - taul) ** 2
                loss12 += (d1_theta12_real - d2_theta12_real) ** 2 + (d1_theta13_real - d2_theta13_real) ** 2  + (d1_theta23_real - d2_theta23_real) ** 2
            if d1idx_fake.sum() > 0 and d2idx_fake.sum() > 0:
                d12_theta11_fake = torch.dot(d1m1_fake, d2m1_fake)
                d12_theta22_fake = torch.dot(d1m2_fake, d2m2_fake)
                d12_theta33_fake = torch.dot(d1m3_fake, d2m3_fake)
                loss12 += (d12_theta11_fake - taus) ** 2 + (d12_theta22_fake - taus) ** 2 + (d12_theta33_fake - taus) ** 2
                loss12 += (d1_theta12_fake - d2_theta12_fake) ** 2 + (d1_theta13_fake - d2_theta13_fake) ** 2 + (d1_theta23_fake - d2_theta23_fake) ** 2
            #print(loss12, loss13, loss23)
            return loss12
        elif d2idx.sum() > 0 and d3idx.sum() > 0:
            if d2idx_real.sum() > 0 and d3idx_real.sum() > 0:
                d23_theta11_real = torch.dot(d2m1_real, d3m1_real)
                d23_theta22_real = torch.dot(d2m2_real, d3m2_real)
                d23_theta33_real = torch.dot(d2m3_real, d3m3_real)
                loss23 += (d23_theta11_real - taul) ** 2 + (d23_theta22_real - taul) ** 2 + (d23_theta33_real - taul) ** 2
                loss23 += (d2_theta12_real - d3_theta12_real) ** 2 + (d2_theta13_real - d3_theta13_real) ** 2 + (d2_theta23_real - d3_theta23_real) ** 2
            if d2idx_fake.sum() > 0 and d3idx_fake.sum() > 0:
                d23_theta11_fake = torch.dot(d2m1_fake, d3m1_fake)
                d23_theta22_fake = torch.dot(d2m2_fake, d3m2_fake)
                d23_theta33_fake = torch.dot(d2m3_fake, d3m3_fake)
                loss23 += (d23_theta11_fake - taus) ** 2 + (d23_theta22_fake - taus) ** 2 + (d23_theta33_fake - taus) ** 2
                loss23 += (d2_theta12_fake - d3_theta12_fake) ** 2 + (d2_theta13_fake - d3_theta13_fake) ** 2 + (d2_theta23_fake - d3_theta23_fake) ** 2
            #print(loss12, loss13, loss23)
            return loss23
        elif d1idx.sum() > 0 and d3idx.sum() > 0:
            if d1idx_real.sum() > 0 and d3idx_real.sum() > 0:
                d13_theta11_real = torch.dot(d1m1_real, d3m1_real)
                d13_theta22_real = torch.dot(d1m2_real, d3m2_real)
                d13_theta33_real = torch.dot(d1m3_real, d3m3_real)
                loss13 += (d13_theta11_real - taul) ** 2 + (d13_theta22_real - taul) ** 2 + (d13_theta33_real - taul) ** 2
                loss13 += (d1_theta12_real - d3_theta12_real) ** 2 + (d1_theta13_real - d3_theta13_real) ** 2 + (d1_theta23_real - d3_theta23_real) ** 2
            if d1idx_fake.sum() > 0 and d3idx_fake.sum() > 0:
                d13_theta11_fake = torch.dot(d1m1_fake, d3m1_fake)
                d13_theta22_fake = torch.dot(d1m2_fake, d3m2_fake)
                d13_theta33_fake = torch.dot(d1m3_fake, d3m3_fake)
                loss13 += (d13_theta11_fake - taus) ** 2 + (d13_theta22_fake - taus) ** 2 + (d13_theta33_fake - taus) ** 2
                loss13 += (d1_theta12_fake - d3_theta12_fake) ** 2 + (d1_theta13_fake - d3_theta13_fake) ** 2 + (d1_theta23_fake - d3_theta23_fake) ** 2
            #print(loss12, loss13, loss23)
            return loss13
        else:
            return 0

class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, features, label, domain, taul=1.0, taus=0.85):
        '''
        :param features: 3 * [B, D]
        :param label: [B,]
        :param domain: [B,]
        :return:
        '''
        m1, m2, m3 = features[0], features[1], features[2]
        d1idx = domain.cuda() == 0
        if d1idx.sum() > 0:
            d1m1, d1m2, d1m3 = m1[d1idx], m2[d1idx], m3[d1idx]
            d1label = label[d1idx]
            d1idx_real = d1label == 1
            d1idx_fake = d1label == 0

            if d1idx_real.sum() > 0:
                d1m1_real = d1m1[d1idx_real] / torch.norm(d1m1[d1idx_real], p=2, dim=1, keepdim=True)
                d1m2_real = d1m2[d1idx_real] / torch.norm(d1m2[d1idx_real], p=2, dim=1, keepdim=True)
                d1m3_real = d1m3[d1idx_real] / torch.norm(d1m3[d1idx_real], p=2, dim=1, keepdim=True)
            if d1idx_fake.sum() > 0:
                d1m1_fake = d1m1[d1idx_fake] / torch.norm(d1m1[d1idx_fake], p=2, dim=1, keepdim=True)
                d1m2_fake = d1m2[d1idx_fake] / torch.norm(d1m2[d1idx_fake], p=2, dim=1, keepdim=True)
                d1m3_fake = d1m3[d1idx_fake] / torch.norm(d1m3[d1idx_fake], p=2, dim=1, keepdim=True)
            if d1idx_real.sum() > 0:
                d1_theta12_real = torch.mm(d1m1_real, d1m2_real.T)
                d1_theta13_real = torch.mm(d1m1_real, d1m3_real.T)
                d1_theta23_real = torch.mm(d1m2_real, d1m3_real.T)
            if d1idx_fake.sum() > 0:
                d1_theta12_fake = torch.mm(d1m1_fake, d1m2_fake.T)
                d1_theta13_fake = torch.mm(d1m1_fake, d1m3_fake.T)
                d1_theta23_fake = torch.mm(d1m2_fake, d1m3_fake.T)

        d2idx = domain.cuda() == 1
        if d2idx.sum() > 0:
            d2m1, d2m2, d2m3 = m1[d2idx], m2[d2idx], m3[d2idx]
            d2label = label[d2idx]
            d2idx_real = d2label == 1
            d2idx_fake = d2label == 0

            if d2idx_real.sum() > 0:
                d2m1_real = d2m1[d2idx_real] / torch.norm(d2m1[d2idx_real], p=2, dim=1, keepdim=True)
                d2m2_real = d2m2[d2idx_real] / torch.norm(d2m2[d2idx_real], p=2, dim=1, keepdim=True)
                d2m3_real = d2m3[d2idx_real] / torch.norm(d2m3[d2idx_real], p=2, dim=1, keepdim=True)
            if d2idx_fake.sum() > 0:
                d2m1_fake = d2m1[d2idx_fake] / torch.norm(d2m1[d2idx_fake], p=2, dim=1, keepdim=True)
                d2m2_fake = d2m2[d2idx_fake] / torch.norm(d2m2[d2idx_fake], p=2, dim=1, keepdim=True)
                d2m3_fake = d2m3[d2idx_fake] / torch.norm(d2m3[d2idx_fake], p=2, dim=1, keepdim=True)
            if d2idx_real.sum() > 0:
                d2_theta12_real = torch.mm(d2m1_real, d2m2_real.T)
                d2_theta13_real = torch.mm(d2m1_real, d2m3_real.T)
                d2_theta23_real = torch.mm(d2m2_real, d2m3_real.T)
            if d2idx_fake.sum() > 0:
                d2_theta12_fake = torch.mm(d2m1_fake, d2m2_fake.T)
                d2_theta13_fake = torch.mm(d2m1_fake, d2m3_fake.T)
                d2_theta23_fake = torch.mm(d2m2_fake, d2m3_fake.T)

        d3idx = domain.cuda() == 2
        if d3idx.sum() > 0:
            d3m1, d3m2, d3m3 = m1[d3idx], m2[d3idx], m3[d3idx]
            d3label = label[d3idx]
            d3idx_real = d3label == 1
            d3idx_fake = d3label == 0
            if d3idx_real.sum() > 0:
                d3m1_real = d3m1[d3idx_real] / torch.norm(d3m1[d3idx_real], p=2, dim=1, keepdim=True)
                d3m2_real = d3m2[d3idx_real] / torch.norm(d3m2[d3idx_real], p=2, dim=1, keepdim=True)
                d3m3_real = d3m3[d3idx_real] / torch.norm(d3m3[d3idx_real], p=2, dim=1, keepdim=True)
            if d3idx_fake.sum() > 0:
                d3m1_fake = d3m1[d3idx_fake] / torch.norm(d3m1[d3idx_fake], p=2, dim=1, keepdim=True)
                d3m2_fake = d3m2[d3idx_fake] / torch.norm(d3m2[d3idx_fake], p=2, dim=1, keepdim=True)
                d3m3_fake = d3m3[d3idx_fake] / torch.norm(d3m3[d3idx_fake], p=2, dim=1, keepdim=True)
            if d3idx_real.sum() > 0:
                d3_theta12_real = torch.mm(d3m1_real, d3m2_real.T)
                d3_theta13_real = torch.mm(d3m1_real, d3m3_real.T)
                d3_theta23_real = torch.mm(d3m2_real, d3m3_real.T)
            if d3idx_fake.sum() > 0:
                d3_theta12_fake = torch.mm(d3m1_fake, d3m2_fake.T)
                d3_theta13_fake = torch.mm(d3m1_fake, d3m3_fake.T)
                d3_theta23_fake = torch.mm(d3m2_fake, d3m3_fake.T)

        loss12, loss13, loss23 = 0, 0, 0
        if d1idx.sum() > 0 and d2idx.sum() > 0 and d3idx.sum() > 0:
            if d1idx_real.sum() > 0 and d2idx_real.sum() > 0 and d3idx_real.sum() > 0:
                d12_theta11_real = torch.mm(d1m1_real, d2m1_real.T)
                d12_theta22_real = torch.mm(d1m2_real, d2m2_real.T)
                d12_theta33_real = torch.mm(d1m3_real, d2m3_real.T)
                d13_theta11_real = torch.mm(d1m1_real, d3m1_real.T)
                d13_theta22_real = torch.mm(d1m2_real, d3m2_real.T)
                d13_theta33_real = torch.mm(d1m3_real, d3m3_real.T)
                d23_theta11_real = torch.mm(d2m1_real, d3m1_real.T)
                d23_theta22_real = torch.mm(d2m2_real, d3m2_real.T)
                d23_theta33_real = torch.mm(d2m3_real, d3m3_real.T)

                loss12 += ((d12_theta11_real - taul) ** 2).sum() + ((d12_theta22_real - taul) ** 2).sum() + (
                        (d12_theta33_real - taul) ** 2).sum()
                loss12 += ((d1_theta12_real.reshape(-1)[:, None] - d2_theta12_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_real.reshape(-1)[:, None] - d2_theta13_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_real.reshape(-1)[:, None] - d2_theta23_real.reshape(-1)[None, :]) ** 2).sum()
                loss13 += ((d13_theta11_real - taul) ** 2).sum() + ((d13_theta22_real - taul) ** 2).sum() + (
                        (d13_theta33_real - taul) ** 2).sum()
                loss13 += ((d1_theta12_real.reshape(-1)[:, None] - d3_theta12_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_real.reshape(-1)[:, None] - d3_theta13_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_real.reshape(-1)[:, None] - d3_theta23_real.reshape(-1)[None, :]) ** 2).sum()
                loss23 += ((d23_theta11_real - taul) ** 2).sum() + ((d23_theta22_real - taul) ** 2).sum() + (
                        (d23_theta33_real - taul) ** 2).sum()
                loss23 += ((d2_theta12_real.reshape(-1)[:, None] - d3_theta12_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta13_real.reshape(-1)[:, None] - d3_theta13_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta23_real.reshape(-1)[:, None] - d3_theta23_real.reshape(-1)[None, :]) ** 2).sum()
            if d1idx_fake.sum() > 0 and d2idx_fake.sum() > 0 and d3idx_fake.sum() > 0:
                d12_theta11_fake = torch.mm(d1m1_fake, d2m1_fake.T)
                d12_theta22_fake = torch.mm(d1m2_fake, d2m2_fake.T)
                d12_theta33_fake = torch.mm(d1m3_fake, d2m3_fake.T)
                d13_theta11_fake = torch.mm(d1m1_fake, d3m1_fake.T)
                d13_theta22_fake = torch.mm(d1m2_fake, d3m2_fake.T)
                d13_theta33_fake = torch.mm(d1m3_fake, d3m3_fake.T)
                d23_theta11_fake = torch.mm(d2m1_fake, d3m1_fake.T)
                d23_theta22_fake = torch.mm(d2m2_fake, d3m2_fake.T)
                d23_theta33_fake = torch.mm(d2m3_fake, d3m3_fake.T)
                loss12 += ((d12_theta11_fake - taus) ** 2).sum() + ((d12_theta22_fake - taus) ** 2).sum() + (
                        (d12_theta33_fake - taus) ** 2).sum()
                loss12 += ((d1_theta12_fake.reshape(-1)[:, None] - d2_theta12_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_fake.reshape(-1)[:, None] - d2_theta13_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_fake.reshape(-1)[:, None] - d2_theta23_fake.reshape(-1)[None, :]) ** 2).sum()
                loss13 += ((d13_theta11_fake - taus) ** 2).sum() + ((d13_theta22_fake - taus) ** 2).sum() + (
                        (d13_theta33_fake - taus) ** 2).sum()
                loss13 += ((d1_theta12_fake.reshape(-1)[:, None] - d3_theta12_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_fake.reshape(-1)[:, None] - d3_theta13_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_fake.reshape(-1)[:, None] - d3_theta23_fake.reshape(-1)[None, :]) ** 2).sum()
                loss23 += ((d23_theta11_fake - taus) ** 2).sum() + ((d23_theta22_fake - taus) ** 2).sum() + (
                        (d23_theta33_fake - taus) ** 2).sum()
                loss23 += ((d2_theta12_fake.reshape(-1)[:, None] - d3_theta12_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta13_fake.reshape(-1)[:, None] - d3_theta13_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta23_fake.reshape(-1)[:, None] - d3_theta23_fake.reshape(-1)[None, :]) ** 2).sum()
            return (loss12 + loss13 + loss23) / 3
        elif d1idx.sum() > 0 and d2idx.sum() > 0:
            if d1idx_real.sum() > 0 and d2idx_real.sum() > 0:
                d12_theta11_real = torch.mm(d1m1_real, d2m1_real.T)
                d12_theta22_real = torch.mm(d1m2_real, d2m2_real.T)
                d12_theta33_real = torch.mm(d1m3_real, d2m3_real.T)
                loss12 += ((d12_theta11_real - taul) ** 2).sum() + ((d12_theta22_real - taul) ** 2).sum() + (
                        (d12_theta33_real - taul) ** 2).sum()
                loss12 += ((d1_theta12_real.reshape(-1)[:, None] - d2_theta12_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_real.reshape(-1)[:, None] - d2_theta13_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_real.reshape(-1)[:, None] - d2_theta23_real.reshape(-1)[None, :]) ** 2).sum()
            if d1idx_fake.sum() > 0 and d2idx_fake.sum() > 0:
                d12_theta11_fake = torch.mm(d1m1_fake, d2m1_fake.T)
                d12_theta22_fake = torch.mm(d1m2_fake, d2m2_fake.T)
                d12_theta33_fake = torch.mm(d1m3_fake, d2m3_fake.T)
                loss12 += ((d12_theta11_fake - taus) ** 2).sum() + ((d12_theta22_fake - taus) ** 2).sum() + (
                        (d12_theta33_fake - taus) ** 2).sum()
                loss12 += ((d1_theta12_fake.reshape(-1)[:, None] - d2_theta12_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_fake.reshape(-1)[:, None] - d2_theta13_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_fake.reshape(-1)[:, None] - d2_theta23_fake.reshape(-1)[None, :]) ** 2).sum()
            return loss12
        elif d2idx.sum() > 0 and d3idx.sum() > 0:
            if d2idx_real.sum() > 0 and d3idx_real.sum() > 0:
                d23_theta11_real = torch.mm(d2m1_real, d3m1_real.T)
                d23_theta22_real = torch.mm(d2m2_real, d3m2_real.T)
                d23_theta33_real = torch.mm(d2m3_real, d3m3_real.T)
                loss23 += ((d23_theta11_real - taul) ** 2).sum() + ((d23_theta22_real - taul) ** 2).sum() + (
                        (d23_theta33_real - taul) ** 2).sum()
                loss23 += ((d2_theta12_real.reshape(-1)[:, None] - d3_theta12_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta13_real.reshape(-1)[:, None] - d3_theta13_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta23_real.reshape(-1)[:, None] - d3_theta23_real.reshape(-1)[None, :]) ** 2).sum()
            if d2idx_fake.sum() > 0 and d3idx_fake.sum() > 0:
                d23_theta11_fake = torch.mm(d2m1_fake, d3m1_fake.T)
                d23_theta22_fake = torch.mm(d2m2_fake, d3m2_fake.T)
                d23_theta33_fake = torch.mm(d2m3_fake, d3m3_fake.T)
                loss23 += ((d23_theta11_fake - taus) ** 2).sum() + ((d23_theta22_fake - taus) ** 2).sum() + (
                        (d23_theta33_fake - taus) ** 2).sum()
                loss23 += ((d2_theta12_fake.reshape(-1)[:, None] - d3_theta12_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta13_fake.reshape(-1)[:, None] - d3_theta13_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d2_theta23_fake.reshape(-1)[:, None] - d3_theta23_fake.reshape(-1)[None, :]) ** 2).sum()

            return loss23
        elif d1idx.sum() > 0 and d3idx.sum() > 0:
            if d1idx_real.sum() > 0 and d3idx_real.sum() > 0:
                d13_theta11_real = torch.mm(d1m1_real, d3m1_real.T)
                d13_theta22_real = torch.mm(d1m2_real, d3m2_real.T)
                d13_theta33_real = torch.mm(d1m3_real, d3m3_real.T)
                loss13 += ((d13_theta11_real - taul) ** 2).sum() + ((d13_theta22_real - taul) ** 2).sum() + (
                        (d13_theta33_real - taul) ** 2).sum()
                loss13 += ((d1_theta12_real.reshape(-1)[:, None] - d3_theta12_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_real.reshape(-1)[:, None] - d3_theta13_real.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_real.reshape(-1)[:, None] - d3_theta23_real.reshape(-1)[None, :]) ** 2).sum()
            if d1idx_fake.sum() > 0 and d3idx_fake.sum() > 0:
                d13_theta11_fake = torch.mm(d1m1_fake, d3m1_fake.T)
                d13_theta22_fake = torch.mm(d1m2_fake, d3m2_fake.T)
                d13_theta33_fake = torch.mm(d1m3_fake, d3m3_fake.T)
                loss13 += ((d13_theta11_fake - taus) ** 2).sum() + ((d13_theta22_fake - taus) ** 2).sum() + (
                        (d13_theta33_fake - taus) ** 2).sum()
                loss13 += ((d1_theta12_fake.reshape(-1)[:, None] - d3_theta12_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta13_fake.reshape(-1)[:, None] - d3_theta13_fake.reshape(-1)[None, :]) ** 2).sum() \
                          + ((d1_theta23_fake.reshape(-1)[:, None] - d3_theta23_fake.reshape(-1)[None, :]) ** 2).sum()
            return loss13
        else:
            return 0
 
