from networks.WFANet import WFANet

def net_factory_3d(net_type="WFANet"):
    if net_type == "WFANet":
        net = WFANet(
               in_channels=1,
               base_filter=16,
               class_num=1,
               embedding_dim=256,
               block_num=12,
               mlp_dim=1024,
               z_idx_list=[3, 6, 9, 12]).cuda()
    else:
        net = None
    return net
