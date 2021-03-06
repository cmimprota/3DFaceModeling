import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from utils import get_vert_connectivity

from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model import Coma
from transform import Normalize
import torch_geometric.transforms as T

# modified from pytorch geometric transformations
class RandomTranslate(object):
    r"""Translates node positions by randomly sampled translation values
    within a given interval. In contrast to other random transformations,
    translation is applied separately at each position.
    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """

    def __init__(self, translate):
        self.translate = translate # here this is 0.1

    def __call__(self, data):
        #print(data)
        #print(self.translate)
        p = torch.rand(1)
        if(p < 0.5):
            rands_x = (-1 + torch.rand(data.x.shape) * (2)) * self.translate # here from -0.1 to 0.1
            data.x = data.x + rands_x

            rands_y = (-1 + torch.rand(data.y.shape) * (2)) * self.translate # here from -0.1 to 0.1
            data.y = data.y + rands_y

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)


class MirrorTranslate(object):
    r"""Mirrors the face horizontally with a certain probability
    """

    def __init__(self, probability, template_mesh, dataset):
        self.probability = probability # here this is 0.5
        self.template_mesh = template_mesh
        self.dataset = dataset

    def __call__(self, data):
        p = torch.rand(1)
        #print(p)
        if(p < self.probability):         
            mesh_verts = data.x
            #print(mesh_verts[0:3])
            mesh_verts[:,1] = mesh_verts[:,1] * (-1)
            #print(mesh_verts[0:3])
            adjacency = get_vert_connectivity(mesh_verts, self.template_mesh.f).tocoo()
            edge_index = torch.LongTensor(np.vstack((adjacency.row, adjacency.col)))
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)
           
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))



def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print(checkpoint_dir)
    visualize = config['visualize']
    output_dir = config['visual_output_dir']
    if visualize is True and not output_dir:
        print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
        output_dir = checkpoint_dir

    if not os.path.exists(output_dir):
        print(output_dir)
        os.makedirs(output_dir)

    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    print("args.split")
    print(args.split)
    normalize_transform = Normalize()
    dataset = ComaDataset(data_dir, 
        dtype='train', 
        split=args.split,
        split_term=args.split_term, 
        pre_transform=normalize_transform
        #, transform=T.Compose([RandomTranslate(0.1),MirrorTranslate(0.5, template_mesh, dataset)])
    )
    if(config['visualize_train'] == False): # only augment data when not measuring
        dataset.transform = T.Compose([RandomTranslate(0.01),MirrorTranslate(0.1, template_mesh, dataset)])
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print('Loading model')
    start_epoch = 1
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)

    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'adagrad':
        coma.to(device)
        optimizer = torch.optim.Adagrad(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    coma.to(device)

    if eval_flag:
        val_loss = evaluate(coma, output_dir, test_loader, dataset_test, template_mesh, device, visualize)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        train_loss = train(coma, train_loader, dataset, optimizer, device, config, template_mesh, epoch)
        val_loss = evaluate(coma, output_dir, test_loader, dataset_test, template_mesh, device, visualize, epoch)
        
        print('epoch ', epoch,' Train loss ', train_loss, ' Val loss ', val_loss)
        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        # early stopping
        # stop when it has not improved the best for 10 epochs
        found_best = False
        if(len(val_loss_history) > 10 and best_val_loss != val_loss):
            for i in range(10):
                if(val_loss_history[-i] != best_val_loss):
                    found_best = True                

            if(found_best == False):
                print("early stopping")
                break

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train(coma, train_loader, dataset, optimizer, device, config, template_mesh, epoch=0):
    coma.train()
    total_loss = 0
    total_vloss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()

        # visualise all of them %1
        if config['visualize_train'] and i % 1 == 0 and epoch % 30 == 0:
            # leave mesh viewer open
            meshviewer = MeshViewers(shape=(1, 2))
            save_out = out.detach().cpu().numpy()
            # batch size of 16
            #print((int(save_out.shape[0])//2319))
            for a in range((int(save_out.shape[0])//2319)):
            #a = 0
                #print(a)
                
                save_out_a = save_out[2319*a:2319*(a+1)]
                yout = (data.y.detach().cpu().numpy())[2319*a:2319*(a+1)]
                save_out_a = save_out_a*dataset.std.numpy()+dataset.mean.numpy()
                expected_out = yout*dataset.std.numpy()+dataset.mean.numpy()
                vloss = np.linalg.norm((save_out_a - expected_out), ord=1) / 2319
                total_vloss += vloss
                #print("vloss train")
                #print(vloss)

                # rotate mesh -90 degrees along x axis
                rotation_matrix = [
                    [1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]
                ]
                result_mesh = Mesh(v=save_out_a, f=template_mesh.f).rotate_vertices(rotation_matrix)
                expected_mesh = Mesh(v=expected_out, f=template_mesh.f).rotate_vertices(rotation_matrix)

                result_mesh.write_obj(os.path.join(config['visual_train_output_dir'], 'result'+str(i)+'.off'))
                expected_mesh.write_obj(os.path.join(config['visual_train_output_dir'], 'expected'+str(i)+'.off'))

                meshviewer[0][0].set_dynamic_meshes([result_mesh])
                meshviewer[0][1].set_dynamic_meshes([expected_mesh])
                meshviewer[0][0].save_snapshot(os.path.join(config['visual_train_output_dir'], 'file'+str(i)+'.png'), blocking=True)
                #a = input()

    mean_vloss = total_vloss / len(dataset) 
    print("mean vloss train")
    print(mean_vloss)
    return total_loss / len(dataset)


def evaluate(coma, output_dir, test_loader, dataset, template_mesh, device, visualize=False, epoch=0):
    coma.eval()
    total_loss = 0
    total_vloss = 0

    #meshviewer = MeshViewers(shape=(1, 2))
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data)
        loss = F.l1_loss(out, data.y)
        #print(loss)
        total_loss += data.num_graphs * loss.item()

        # visualise all of them %1
        if visualize and i % 1 == 0 and epoch % 30 == 0:
            # leave mesh viewer open
            meshviewer = MeshViewers(shape=(1, 2))
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()
            # rotate mesh -90 degrees along x axis
            rotation_matrix = [
                [1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0]
            ]
            result_mesh = Mesh(v=save_out, f=template_mesh.f).rotate_vertices(rotation_matrix)
            expected_mesh = Mesh(v=expected_out, f=template_mesh.f).rotate_vertices(rotation_matrix)
            vloss = np.linalg.norm((save_out - expected_out), ord=1) / 2319
            total_vloss += vloss
            #print("vloss test")
            #print(vloss)

            result_mesh.write_obj(os.path.join(output_dir, 'result'+str(i)+'.off'))
            expected_mesh.write_obj(os.path.join(output_dir, 'expected'+str(i)+'.off'))

            meshviewer[0][0].set_dynamic_meshes([result_mesh])
            meshviewer[0][1].set_dynamic_meshes([expected_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(output_dir, 'file'+str(i)+'.png'), blocking=True)
            #a = input()

    mean_vloss = total_vloss / len(dataset)
    print("mean vloss test")
    print(mean_vloss)
    return total_loss / len(dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
