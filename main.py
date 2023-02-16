import torch
import torch.nn.functional as F
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from BWGNN import *
from sklearn.model_selection import train_test_split
import pickle as pkl


def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    if dataset_name in ['tfinance', 'tsocial']:
        index = list(range(len(labels)))
        if dataset_name == 'amazon':
            index = list(range(3305, len(labels)))

        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=args.train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=0.67,
                                                                random_state=2, shuffle=True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1
    else:
        train_mask = torch.ByteTensor(g.ndata['train_mask'])
        val_mask = torch.ByteTensor(g.ndata['val_mask'])
        test_mask = torch.ByteTensor(g.ndata['test_mask'])
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    best_loss = 100

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(1, args.epoch+1):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        loss = F.cross_entropy(logits[val_mask], labels[val_mask], weight=torch.tensor([1., weight]))
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

        if loss <= best_loss:
            best_loss = loss
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            pred_y = probs
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))
        if args.del_ratio == 0 and e % 20 == 0:
            with open(f'probs_{dataset_name}_BWGNN_{e}_{args.homo}.pkl', 'wb') as f:
                pkl.dump(pred_y, f)

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    result = 'REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100)
    with open('result.txt', 'a+') as f:
        f.write(f'{result}\n')
    return final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--del_ratio", type=float, default=0., help="delete ratios")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--load_epoch", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")

    args = parser.parse_args()
    # with open('result.txt', 'a+') as f:
    #     f.write(f'{args}\n')
    print(args)
    dataset_name = args.dataset
    del_ratio = args.del_ratio
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    adj_type = args.adj_type
    load_epoch = args.load_epoch
    data_path = args.data_path
    graph = Dataset(load_epoch, dataset_name, del_ratio, homo, data_path, adj_type=adj_type).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    # official seed
    set_random_seed(717)

    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        result = 'MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs))
        print(result)

