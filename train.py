import torch
import data_utilize1 as data1
import data_utilize2 as data2
import data_utilize3 as data3
import data_utilize4 as data4
import torch.utils.data as Datas
import Network as Network
import metrics as criterion
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
import os
import gc
import time


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device_ids = [0]
device = torch.device("cuda:0")

for j in range (0,16):
    label_channel = 1
    for i in range(1, 5):
        time.sleep(20)
        ##########初始化网络 加载数据
        net_server = Network.Unet3D(1, 16).to(device)
        if i==1:
            data = data1.train_data
            net_client_old = Network.Unet3D(1, 16).to(device)
            net_client_new = Network.Unet3D(1, 16).to(device)
        elif i==2:
            data = data2.train_data
            net_client_old = Network.Unet3D(1, 16).to(device)
            net_client_new = Network.Unet3D(1, 16).to(device)
        elif i==3:
            data = data3.train_data
            net_client_old = Network.Unet3D(1, 16).to(device)
            net_client_new = Network.Unet3D(1, 16).to(device)
        else:
            data = data4.train_data
            net_client_old = Network.Unet3D(1, 16).to(device)
            net_client_new = Network.Unet3D(1, 16).to(device)

        dataloder = Datas.DataLoader(dataset=data, batch_size=1, shuffle=True)

        ##加载服务器平均模型
        with torch.no_grad():
            flag_fed = os.path.exists('./pkl/sever.pkl')
            if flag_fed == True:
                load_server = torch.load('./pkl/sever.pkl')
                net_server_dict = net_server.state_dict()
                net_client_new.load_state_dict(model_dict)
                print("loading-Server-to-client-" + str(i) + ",success")


        ###########加载old模型
        flag_loc = os.path.exists('./pkl/Client-old' + str(i) + '.pkl')
        if flag_loc == True:
            previous_client = torch.load('./pkl/Client-old' + str(i) + '.pkl')
            model_dict = net_client_old.state_dict()
            previous_client = {k: v for k, v in previous_client.items() if k in model_dict}
            model_dict.update(previous_client)
            net_client_old.load_state_dict(model_dict)
            print("loading-old-client-" + str(i) + ",success")

        opt = torch.optim.Adam(net_client_new.parameters(), lr=1e-4)
        opt1 = torch.optim.Adam(net_client_old.parameters(), lr=1e-4)

        criterion_CE = criterion.crossentry()
        criterion_dice = criterion.DiceMeanLoss1()
        criterion_kl = torch.nn.KLDivLoss()
        criterion_mse = torch.nn.MSELoss()

        # label_channel = np.random.randint(1, high=16)


        stps = 200
        for epoch in range(stps):
            for step, (image3_norm, label, name_img) in enumerate(dataloder):
                image3 = image3_norm.to(device).float()
                label = label.to(device).float()
                b, c, z, w, h = image3.shape
                # print(image3.shape)

                with torch.no_grad():
                    old_seg = net_client_old(image3)

                    old_label = torch.zeros(b, 16, z, w, h)
                    old_label[:, 0:1, :, :, :] = label[:, 0:1, :, :, :]
                    old_label[:, 1:, :, :, :] = old_seg[:, 1:, :, :, :]
                    old_label[:, label_channel:label_channel + 1, :, :, :] = label[:, label_channel:label_channel + 1, :, :, :]


                new_seg = net_client_new(image3)

                # ones = torch.ones(b, 1, z, w, h)
                #
                # new_label = label[:,label_channel:label_channel+1,:,:,:]
                # ones = ones - new_label
                # new_label = torch.cat((ones, new_label), 1)

                loss = criterion_dice(new_seg, old_label.to(device)) + criterion_CE(new_seg, old_label.to(device))

                opt.zero_grad()
                loss.backward()
                opt.step()

                torch.save(net_client_new.state_dict(), './pkl/Iter' + str(j) + 'Client' + str(i) + '.pkl')
                torch.save(net_client_new.state_dict(), './pkl/Client-old' + str(i) + '.pkl')
                torch.save(net_client_new.state_dict(), './pkl/Client' + str(i) + '.pkl')

                print('Iter:', j, 'Client:', i, 'EPOCH:', epoch, '|Step:', step, '|loss:', loss.data.cpu().numpy())





                pt = image3[0, 0, :, :, :].data.cpu().numpy()
                out1 = sitk.GetImageFromArray(pt)
                sitk.WriteImage(out1, './state/imge.nii')
                mm1 = torch.zeros(z, w, h)
                mm2 = torch.zeros(z, w, h)
                mm3 = torch.zeros(z, w, h)
                for mj in range(16):
                    mm1 = mm1 + label[0, mj, :, :, :].data.cpu() * mj
                    mm2 = mm2 + old_seg[0, mj, :, :, :].data.cpu() * mj
                    mm3 = mm3 + new_seg[0, mj, :, :, :].data.cpu() * mj

                pt1 = mm1.data.cpu().numpy()
                pt2 = mm2.data.cpu().numpy()
                pt3 = mm3.data.cpu().numpy()
                out1 = sitk.GetImageFromArray(pt1)
                sitk.WriteImage(out1, './state/label.nii')
                out2 = sitk.GetImageFromArray(pt2)
                sitk.WriteImage(out2, './state/old_seg.nii')
                out3 = sitk.GetImageFromArray(pt3)
                sitk.WriteImage(out3, './state/new_seg.nii')

##############
    # del net_server
    # del net_client
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        C1 = torch.load('./pkl/Client1.pkl')
        C2 = torch.load('./pkl/Client2.pkl')
        C3 = torch.load('./pkl/Client3.pkl')
        C4 = torch.load('./pkl/Client4.pkl')


        C_avg = C1

        for k, v in C1.items():
            layer_name = k
            split_str = k.split(".")
            if split_str[0] != "out":
                if split_str[1]!="finat":

                    layer_w1 = C1[k].data
                    layer_w2 = C2[k].data
                    layer_w3 = C3[k].data
                    layer_w4 = C4[k].data

                    C_avg[k].data = (layer_w1 + layer_w2 + layer_w3 + layer_w4) / 4

        torch.save(C_avg, './pkl/sever.pkl')
        torch.save(C_avg, './pkl/Iter' + str(j) + 'sever.pkl')

        del C1
        del C2
        del C3
        del C4
        del C_avg
        gc.collect()
        torch.cuda.empty_cache()







