import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
from utils.DataHelper import dense_to_coo

#def ShowGraph(graph, nodeLabel, EdgeLabel):
def ShowGraph(graph, nodeLabel):
    plt.figure(figsize=(8, 8))
    #G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split()) #转换dgl graph to networks
    G = graph.to_networkx(node_attrs=nodeLabel.split())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="black", node_size=500, with_labels=True)  #画图，设置节点大小

    node_data = nx.get_node_attributes(G, nodeLabel)  #获取节点的desc属性
    #重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2}
    node_labels = {index: "N:" + str(data) for index, data in enumerate(node_data)}

    pos_higher = {}
    for k, v in pos.items():  #调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if (v[1] > 0):
            pos_higher[k] = (v[0]-0.04, v[1]+0.04)
        else:
            pos_higher[k] = (v[0]-0.04, v[1]-0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)  #将desc属性，显示在节点上


    '''edge_labels = nx.get_edge_attributes(G, EdgeLabel)  #获取边的weights属性，
    edge_labels = {(key[0], key[1]): "w:" + str(edge_labels[key].item()) for key in
                   edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上

    print(G.edges.data())'''
    plt.show()


#detect_degree(list for graphs)
def detect_degree(graphs):
    for i in range(len(graphs)):
        buf = []
        for node in graphs[i].nodes():
            #undirected graph: in_degrees=out_degrees
            if graphs[i].in_degrees(node) == 0:
               buf.append(int(node))
        if buf != []:
            #print(buf)
            graphs[i] = dgl.remove_nodes(graphs[i], th.tensor(buf))

'''
#detach features, e_features, e_adj, adj, t_mat
def BDgraph_info(batch_graph):
    inc_mat = batch_graph.incidence_matrix(typestr='both')
    dense_inc_mat = inc_mat.to_dense()
    t_mat = th.abs(dense_inc_mat)
    sparse_t_mat = dense_to_coo(t_mat)

    adj = batch_graph.adj()
    features = batch_graph.ndata['x']

    #convert to line graph
    line_graphs = dgl.line_graph(batch_graph)
    e_adj = line_graphs.adj()
    e_features = batch_graph.edata['w']

    return features, e_features, e_adj, adj, sparse_t_mat
'''

#readout features in terms of a drug graph
def BDgraph_readout(batch_size, batch_graph, batch_features):
    tensor_list = []
    #determine the size of last batch
    batch_size_buf = batch_size
    if (batch_graph.batch_size%batch_size) != 0:
        batch_size_buf = batch_graph.batch_size

    for i in range(batch_size_buf):
        '''print("i: {}".format(i))'''
        start_row = 0
        #when i=0, unable to excute 'for j in range(i)'
        #'for j in range(i)': get start row
        for j in range(i):
            '''print("j: {}".format(j))'''
            row_num = dgl.slice_batch(batch_graph, j).num_nodes()
            start_row = start_row + row_num

        ith_rowNum = dgl.slice_batch(batch_graph, i).num_nodes()
        '''print("rowNum: {}".format(ith_rowNum))'''
        ith_graph = batch_features[start_row:(start_row+ith_rowNum), :]
        '''print("rowNum: {}".format(ith_graph.size()))'''
        #dim=0, calculate sum/mean in terms of column
        readout_ith_Graph = th.mean(ith_graph, dim=0)
        tensor_list.append(readout_ith_Graph)

    tensor_graphs = th.stack(tensor_list)
    return tensor_graphs
