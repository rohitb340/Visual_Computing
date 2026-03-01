import torch


def fedevi_scoring_func(global_model, local_model, dataloader):
    global_model.eval()
    global_model.cuda()
    local_model.eval()
    local_model.cuda()

    udis_list = torch.tensor([]).cuda()
    udata_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for (image, label) in dataloader:
            image = image.cuda()
            # surrogate global model
            g_logit = global_model(image)
            g_logit = torch.clamp_max(g_logit, 80)
            alpha = torch.exp(g_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)

            g_pred = alpha / total_alpha
            g_entropy = torch.sum(- g_pred * torch.log(g_pred), dim=1)     
            g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            
            g_u_dis = g_entropy - g_u_data
            udis_list = torch.cat((udis_list, g_u_dis.mean(dim=[1,2])))

            l_logit = local_model(image)
            l_logit = torch.clamp_max(l_logit, 80)
            alpha = torch.exp(l_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)
            l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

            udata_list = torch.cat((udata_list, l_u_data.mean(dim=[1,2])))

    return udis_list.mean(), udata_list.mean()