import os
import json
import logging
import torch as th
import numpy as np
from datetime import datetime
from dgl.data.utils import load_graphs
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader
from utils.DglOperator import detect_degree
from utils.Pharmacophore import get_pharmacophore
from utils.DataHelper import LoadData, LoadData2, GetSamples, GetSamples1
from utils.DataHelper import LabelDT, Shuttle, Shuttle_pharma, batch_normalize
from utils.Emetrics import get_MSE, get_cindex, get_rm2, get_RMSE, get_spearmanr
from cpu import ConfigArgumentParser, EvalHook, Trainer, save_args, set_random_seed, setup_logger
from model.GraphModel import DatasetIterater, DatasetIteraterP, BatchPad, Pcollate, Gcollate, GraphBasedModel

logger = logging.getLogger(__name__)
def parse_args():
    parser = ConfigArgumentParser(description="Affinity prediction")

    parser.add_argument("--log_dir", type=str, default="./log_out", help="Directory to save checkpoints and logs.")
    parser.add_argument("--work_dir", type=str, default="./", help="Directory to save runtime_config.yaml.")

    parser.add_argument("--davis_dir", type=str, default="./data/davis/", help="Directory to save dataset.")
    parser.add_argument("--kiba_dir", type=str, default="./data/kiba/", help="...")
    parser.add_argument("--fdavis_dir", type=str, default="./data/filtered davis/", help="...")
    parser.add_argument("--metz_dir", type=str, default="./data/metz/", help="...")
    parser.add_argument("--toxcast_dir", type=str, default="./data/toxcast/", help="...")

    parser.add_argument("--davis", type=str, default='davis', help="Dataset name.")
    parser.add_argument("--kiba", type=str, default='kiba', help="...")
    parser.add_argument("--fdavis", type=str, default='filtered davis', help="...")
    parser.add_argument("--metz", type=str, default='metz', help="...")
    parser.add_argument("--toxcast", type=str, default='toxcast', help="...")

    parser.add_argument("--seed", type=int, default=3, help="Random seed, set to negative to randomize everything.")
    parser.add_argument("--deterministic", action="store_true", help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--smi_maxDA", type=int, default=85, help="Limit max length of smiles in davis.")
    parser.add_argument("--tar_maxDA", type=int, default=1200, help="Limit max length of targets in davis.")
    parser.add_argument("--smi_maxKB", type=int, default=100, help="...")
    parser.add_argument("--tar_maxKB", type=int, default=1000, help="...")
    parser.add_argument("--smi_maxFDA", type=int, default=85, help="...")
    parser.add_argument("--tar_maxFDA", type=int, default=1200, help="...")
    parser.add_argument("--smi_maxMT", type=int, default=100, help="...")
    parser.add_argument("--tar_maxMT", type=int, default=1200, help="...")
    parser.add_argument("--smi_maxTC", type=int, default=100, help="...")
    parser.add_argument("--tar_maxTC", type=int, default=1200, help="...")

    parser.add_argument("--train_numDA", type=int, default=25046, help="total 30056")
    parser.add_argument("--test_numDA", type=int, default=5010, help="...")
    parser.add_argument("--train_numKB", type=int, default=98545, help="total 118254")
    parser.add_argument("--test_numKB", type=int, default=19709, help="...")
    parser.add_argument("--train_numFDA", type=int, default=7605, help="total 9125")
    parser.add_argument("--test_numFDA", type=int, default=1520, help="...")
    parser.add_argument("--train_numMT", type=int, default=28207, help="total 35259")
    parser.add_argument("--test_numMT", type=int, default=7052, help="...")
    parser.add_argument("--train_numTC", type=int, default=80214, help="total 114589")
    parser.add_argument("--test_numTC", type=int, default=34375, help="...")

    parser.add_argument("--device", type=str, default='cuda:0', help="Device to train on cuda:0.")
    parser.add_argument("--batch_size", type=int, default=16, help="Input batch size for training.")
    parser.add_argument("--batch_num", type=int, default=476, help="Number of batch (3080 in KIBA, 783 in davis)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train(300/600/XXX).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Learning rate")
    parser.add_argument("--min_train_loss", type=float, default=1e6, help="Initial minimum epoch train loss")
    parser.add_argument("--grad_step", type=int, default=8, help="Gradient accumulate(128/8/XXXX)")
    parser.add_argument("--log_iter", type=int, default=70, help="Interval for logging to console and tensorboard")
    #model config
    parser.add_argument("--model_config", type=str, default="./model/config.json", help="Initial model parameters")

    return parser.parse_args()


def build_dataset(dir, dataset_name, dru_limit, tar_limit, train_num, test_num):
    #logspance_trans = True if davis else False(kiba)
    '''drug, target, affi_matrix = LoadData(dir, logspance_trans=True)'''
    drug_info, target_info, affi = LoadData2(dir, logspance_trans=None)
    drug_graphs, _ = load_graphs(dir+dataset_name + "_DrugGraphs.bin")
    target_graphs, _ = load_graphs(dir+dataset_name+"_TargetGraphs_8A.bin")
    #delete 0-degree nodes for graphs
    detect_degree(drug_graphs)
    detect_degree(target_graphs)
    #DO.ShowGraph(drug_graphs[1886],"x")
    '''Gdrugs, Gtargets, Glabels = GetSamples(dataset_name, drug_graphs, target_graphs, affi_matrix)
    Sdrugs, Stargets, Slabels = GetSamples(dataset_name, drug, target, affi_matrix)'''
    Gdrugs, Gtargets, Glabels, Sdrugs, Stargets, Slabels = GetSamples1(dataset_name, drug_info, drug_graphs,
                                                                           target_info, target_graphs, affi)
    #shuttle samples (len(Glabels)  or len(Slabels))
    index = [i for i in range(len(Glabels))]
    np.random.shuffle(index)

    Gdrugs_shuttled, Gtargets_shuttled, Glabels_shuttled = Shuttle(Gdrugs, Gtargets, Glabels, index)
    LSdrugs, LStargets = LabelDT(Sdrugs, Stargets, dru_limit, tar_limit)
    LSdrugs_shuttled, LStargets_shuttled, Slabels_shuttled = Shuttle(LSdrugs, LStargets, Slabels, index)
    drug_pharma = get_pharmacophore(Sdrugs)
    drugPharma_shuttled = Shuttle_pharma(drug_pharma, index)

    Gtrain_iter = DatasetIterater(Gdrugs_shuttled[0:train_num], Gtargets_shuttled[0:train_num],
                                                                                      Glabels_shuttled[0:train_num])
    Strain_iter = DatasetIterater(LSdrugs_shuttled[0:train_num], LStargets_shuttled[0:train_num],
                                                                                      Slabels_shuttled[0:train_num])
    Ptrain_iter = DatasetIteraterP(drugPharma_shuttled[0:train_num])

    Gtest_iter = DatasetIterater(Gdrugs_shuttled[train_num:train_num+test_num],
                    Gtargets_shuttled[train_num:train_num+test_num], Glabels_shuttled[train_num:train_num+test_num])
    Stest_iter = DatasetIterater(LSdrugs_shuttled[train_num:train_num+test_num],
                   LStargets_shuttled[train_num:train_num+test_num], Slabels_shuttled[train_num:train_num+test_num])
    Ptest_iter = DatasetIteraterP(drugPharma_shuttled[train_num:train_num+test_num])

    return Gtrain_iter, Strain_iter, Ptrain_iter, Gtest_iter, Stest_iter, Ptest_iter


def build_dataloader(args):
    Gtrain_iter, Strain_iter, Ptrain_iter, Gtest_iter, Stest_iter, Ptest_iter = build_dataset(args.fdavis_dir,
                           args.fdavis, args.smi_maxFDA, args.tar_maxFDA, args.train_numFDA, args.test_numFDA)
    #GDloader and SDloader params:  drop_last=False, shuffle=False
    GDtrain_loader = GraphDataLoader(Gtrain_iter, batch_size=args.batch_size, collate_fn=Gcollate)
    SDtrain_loader = DataLoader(Strain_iter, batch_size=args.batch_size, collate_fn=BatchPad)
    PDtrain_loader = DataLoader(Ptrain_iter, batch_size=args.batch_size, collate_fn=Pcollate)
    GDtest_loader = GraphDataLoader(Gtest_iter, batch_size=args.batch_size, collate_fn=Gcollate)
    SDtest_loader = DataLoader(Stest_iter, batch_size=args.batch_size, collate_fn=BatchPad)
    PDtest_loader = DataLoader(Ptest_iter, batch_size=args.batch_size, collate_fn=Pcollate)

    return GDtrain_loader, SDtrain_loader, PDtrain_loader, GDtest_loader, SDtest_loader, PDtest_loader


def main(args):
    #2. Basic setup
    setup_logger(output_dir=args.log_dir)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    #If args.seed is negative or None, will use a randomly generated seed
    set_random_seed(args.seed, args.deterministic)
    #3. Create data_loader, model, optimizer, lr_scheduler
    GDtrain, SDtrain, PDtrain, GDtest, SDtest, PDtest = build_dataloader(args)

    model_condig = json.load(open(args.model_config, 'r'))
    Gmodel = GraphBasedModel(model_condig).to(args.device)
    optim = th.optim.Adam(params=Gmodel.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    reg_loss_fn = th.nn.MSELoss()

    epoach_loss = []
    for epoch in range(args.epochs):
        print("===========================Go for Training====================================")
        #start batch normalization and dropout
        Gmodel.train()
        for batch_id, (Batch_GDdata, Batch_SDdata, Batch_PDdata) in enumerate(zip(GDtrain, SDtrain, PDtrain)):
            BDgraph, BTgraph, real_affi = Batch_GDdata[0], Batch_GDdata[1], Batch_GDdata[2]
            nor_d_features, nor_de_features, nor_de_adj, nor_d_adj, d_mat = batch_normalize(BDgraph)

            out_pred_affi = Gmodel(BDgraph.to(args.device), nor_d_features.to(args.device),
                                   nor_de_features.to(args.device), nor_de_adj.to(args.device),
                                   nor_d_adj.to(args.device), d_mat.to(args.device), Batch_PDdata.to(args.device),
                                   BTgraph.to(args.device), BTgraph.ndata['x'].to(args.device),
                                   BTgraph.edata['w'].to(args.device), Batch_SDdata[0].to(args.device),
                                   Batch_SDdata[1].to(args.device), args.batch_size)
            pred_affi = out_pred_affi.squeeze(1)
            loss = reg_loss_fn(pred_affi, real_affi.to(args.device))
            loss.backward()

            if (batch_id+1) % args.grad_step == 0:
                optim.step()
                optim.zero_grad()

            if (batch_id+1) % args.log_iter == 0:
                print("Training at epoch: {}, step: {}, loss is: {}".format(epoch+1, batch_id,
                                                                              loss.cpu().detach().numpy()))
            if (batch_id+1) == args.batch_num:
                epoach_loss.append(loss.cpu().detach().numpy())
                print('\n')
                print("Training after epoch: {}, loss is: {}".format(epoch+1, loss.cpu().detach().numpy()))

        if epoach_loss[epoch] < args.min_train_loss:
            args.min_train_loss = epoach_loss[epoch]
            th.save(Gmodel.state_dict(), './final_model.pth')
            print("Save best model after epoch: {}".format(epoch+1))

        scheduler.step()

    np.savetxt('./epoach_loss.csv', epoach_loss, delimiter=',')
    #th.save(Gmodel.state_dict(),'./final_model.pth')
    with th.no_grad():
        print("===========================Go for testing====================================")
        train_obs, train_pred = [], []
        test_obs, test_pred = [], []
        pred_Gmodel = GraphBasedModel(model_condig).to(args.device)
        #keep Batch Normalization and avoid Dropout
        pred_Gmodel.eval()
        pred_Gmodel.load_state_dict(th.load('./final_model.pth'))
        #train data
        for Batch_GDdata, Batch_SDdata, Batch_PDdata in zip(GDtrain, SDtrain, PDtrain):
            BDgraph, BTgraph, real_affi = Batch_GDdata[0], Batch_GDdata[1], Batch_GDdata[2]
            nor_d_features, nor_de_features, nor_de_adj, nor_d_adj, d_mat = batch_normalize(BDgraph)

            out_eval_affi = pred_Gmodel(BDgraph.to(args.device), nor_d_features.to(args.device),
                                        nor_de_features.to(args.device), nor_de_adj.to(args.device),
                                        nor_d_adj.to(args.device), d_mat.to(args.device), Batch_PDdata.to(args.device),
                                        BTgraph.to(args.device), BTgraph.ndata['x'].to(args.device),
                                        BTgraph.edata['w'].to(args.device), Batch_SDdata[0].to(args.device),
                                        Batch_SDdata[1].to(args.device), args.batch_size)
            eval_affi = out_eval_affi.squeeze(1)
            train_obs.extend(real_affi.tolist())
            train_pred.extend(eval_affi.tolist())

        #test data
        for Batch_GDdata, Batch_SDdata, Batch_PDdata in zip(GDtest, SDtest, PDtest):
            BDgraph, BTgraph, real_affi = Batch_GDdata[0], Batch_GDdata[1], Batch_GDdata[2]
            nor_d_features, nor_de_features, nor_de_adj, nor_d_adj, d_mat = batch_normalize(BDgraph)

            out_eval_affi = pred_Gmodel(BDgraph.to(args.device), nor_d_features.to(args.device),
                                        nor_de_features.to(args.device), nor_de_adj.to(args.device),
                                        nor_d_adj.to(args.device), d_mat.to(args.device), Batch_PDdata.to(args.device),
                                        BTgraph.to(args.device), BTgraph.ndata['x'].to(args.device),
                                        BTgraph.edata['w'].to(args.device), Batch_SDdata[0].to(args.device),
                                        Batch_SDdata[1].to(args.device), args.batch_size)
            eval_affi = out_eval_affi.squeeze(1)
            test_obs.extend(real_affi.tolist())
            test_pred.extend(eval_affi.tolist())

        np.savetxt('test_obs.csv', test_obs, delimiter=',')
        np.savetxt('test_pred.csv', test_pred, delimiter=',')

    print("===========================Go for emetrics====================================")
    '''print("train-> mse:{:.3f}, ci:{:.3f}, rm2:{:.3f}".format(get_MSE(train_obs, train_pred),
                                            get_cindex(train_obs, train_pred), get_rm2(train_obs, train_pred)))
    print("test-> mse:{:.3f}, ci:{:.3f}, rm2:{:.3f}".format(get_MSE(test_obs, test_pred),
                                                get_cindex(test_obs, test_pred), get_rm2(test_obs, test_pred)))'''
    print("train-> rmse:{:.3f}, ci:{:.3f}, Spearman:{:.3f}".format(get_RMSE(train_obs, train_pred),
                         get_cindex(train_obs, train_pred), get_spearmanr(train_obs, train_pred)))
    print("test-> rmse:{:.3f}, ci:{:.3f}, Spearman:{:.3f}".format(get_RMSE(test_obs, test_pred),
                          get_cindex(test_obs, test_pred), get_spearmanr(test_obs, test_pred)))


if __name__ == '__main__':
    #1. Create an argument parser supporting loading YAML configuration file
    args = parse_args()
    beginT = datetime.now()
    print("Starting Time: {}".format(beginT.strftime(r'%m-%d-%H:%M:%S')))
    main(args)
    endT = datetime.now()
    print("Ending Time: {}".format(endT.strftime(r'%m-%d-%H:%M:%S')))
    interval = endT - beginT
    m, s = divmod(interval.total_seconds(), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Duration is {}d {}h {}m {}s".format(d, h, m, s))
