"""Study RGNN architecture with an explicit edge feature dimension."""
import torch
from torch import nn
from torch.nn import functional as F

class RGNN(nn.Module):
    def __init__(self, din, rels, hidden=64, layers=2, drop=0.3, edge_dim=9, tasks=None, message_mode='legacy'):
        super().__init__()
        if message_mode not in ('legacy','temporal_difference.v1'):
            raise ValueError('Unknown RGNN message mode')
        self.message_mode=message_mode
        self.rels = list(rels); self.drop = drop
        dims = [din] + [hidden] * layers
        self.self_w = nn.ModuleList([nn.Linear(dims[k], dims[k + 1]) for k in range(layers)])
        self.rel_w = nn.ModuleList([nn.ModuleDict({r: nn.Linear(dims[k], dims[k + 1], bias=False) for r in self.rels}) for k in range(layers)])
        self.edge_w = nn.ModuleList([nn.ModuleDict({r: nn.Linear(edge_dim, dims[k + 1], bias=False) for r in self.rels}) for k in range(layers)])
        self.tasks = tuple(tasks) if tasks is not None else None
        def head():
            return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 1))
        if self.tasks is None:
            self.head = head()
        else:
            if not self.tasks or len(set(self.tasks)) != len(self.tasks):
                raise ValueError('RGNN tasks must be nonempty and unique')
            self.heads = nn.ModuleDict({task:head() for task in self.tasks})
        temporal=[r for r in self.rels if r.startswith('t_')] if message_mode!='legacy' else []
        # Keep original encoder/head initialization and legacy state keys intact.
        self.difference_w=nn.ModuleList([nn.ModuleDict({r:nn.Linear(2*dims[k],hidden,bias=False)
            for r in temporal}) for k in range(layers)])
        self.gate_w=nn.ModuleList([nn.ModuleDict({r:nn.Sequential(
            nn.Linear(2*dims[k]+edge_dim,32),nn.ReLU(),nn.Linear(32,1)) for r in temporal}) for k in range(layers)])
        self.match_count_w=nn.ModuleList([nn.ModuleDict({r:nn.Linear(2,hidden,bias=False)
            for r in temporal}) for k in range(layers)])

    def forward(self, x, R):
        h = x
        for k in range(len(self.self_w)):
            out = self.self_w[k](h)
            for r in self.rels:
                (ei, ea) = R[r]
                msg = self.rel_w[k][r](h[ei[0]]) + self.edge_w[k][r](ea)
                if r in self.difference_w[k]:
                    delta=h[ei[0]]-h[ei[1]]
                    differences=torch.cat((delta,delta.abs()),dim=1)
                    gate=torch.sigmoid(self.gate_w[k][r](torch.cat((differences,ea),dim=1)))
                    msg=(msg+self.difference_w[k][r](differences))*gate
                agg = torch.zeros_like(out).index_add_(0, ei[1], msg)
                counts = torch.zeros(out.shape[0], device=out.device).index_add_(0, ei[1], torch.ones(ei.shape[1], device=out.device))
                out = out + agg / counts.clamp(min=1)[:, None]
                if r in self.match_count_w[k]:
                    out=out+self.match_count_w[k][r](torch.stack(((counts>0).to(h.dtype),torch.log1p(counts)),dim=1))
            h = F.dropout(F.relu(out), self.drop, self.training)
        if self.tasks is None:
            return self.head(h).squeeze(-1)
        return torch.cat([self.heads[task](h) for task in self.tasks],dim=1)


class GraphTaskExperts(nn.Module):
    """Route declared tasks through complete RGNNs using one verified input schema."""
    def __init__(self,base,experts,owners):
        super().__init__()
        self.base=base
        self.experts=nn.ModuleList(experts)
        self.owners=tuple(owners)

    def forward(self,x,relations):
        outputs=[self.base(x,relations)]+[model(x,relations) for model in self.experts]
        return torch.cat([outputs[owner][:,j:j+1] for j,owner in enumerate(self.owners)],dim=1)


def model_from_settings(din,relations,settings,tasks):
    """Ordinary checkpoints retain exact RGNN state keys; experts are explicit."""
    configs=settings.get('task_experts',[])
    if not isinstance(configs,list):raise ValueError('GNN task experts must be a list')
    task_order=tuple(tasks) if tasks is not None else ()
    owners=[0]*len(task_order)
    for number,config in enumerate(configs,1):
        if not isinstance(config,dict) or set(config)!={'tasks','message_mode'}:
            raise ValueError('GNN task expert needs tasks and message_mode')
        names=config['tasks']
        if not isinstance(names,list) or not names:
            raise ValueError('GNN task expert must own a nonempty list of tasks')
        for task in names:
            if not isinstance(task,str) or task not in task_order:
                raise ValueError('Unknown GNN expert task')
            index=task_order.index(task)
            if owners[index]:raise ValueError('Duplicate GNN expert task owner')
            owners[index]=number
    def create(mode):
        return RGNN(din,relations,hidden=settings['hidden'],edge_dim=settings['edge_dim'],
            tasks=tasks,message_mode=mode)
    base=create(settings.get('message_mode','legacy'))
    if not configs:return base
    experts=[create(config['message_mode']) for config in configs]
    return GraphTaskExperts(base,experts,owners)
