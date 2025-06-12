import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class SemanticConsistencyGNN(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.node_encoder = nn.Linear(3, 64)
        self.gnn_layers = nn.ModuleList([
            gnn.GCNConv(64, 128),
            gnn.GCNConv(128, 256),
            gnn.GCNConv(256, 128)
        ])
        self.global_pool = gnn.global_mean_pool
        
        self.fusion = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def create_face_graph(self, landmarks):
        batch_graphs = []
        
        for batch_idx in range(landmarks.shape[0]):
            landmark_coords = landmarks[batch_idx].view(-1, 3)
            
            # Create edges between nearby landmarks
            edges = []
            for i in range(len(landmark_coords)):
                for j in range(i+1, min(i+10, len(landmark_coords))):
                    edges.append([i, j])
                    edges.append([j, i])  # Add both directions for undirected graph
            
            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            graph = Data(x=landmark_coords, edge_index=edge_index.to(landmarks.device))
            batch_graphs.append(graph)
        
        return batch_graphs
    
    def forward(self, landmarks):
        batch_graphs = self.create_face_graph(landmarks)
        batch_features = []
        
        for graph in batch_graphs:
            x = self.node_encoder(graph.x)
            
            for gnn_layer in self.gnn_layers:
                x = F.relu(gnn_layer(x, graph.edge_index))
            
            graph_feature = self.global_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            batch_features.append(graph_feature)
        
        batch_tensor = torch.stack(batch_features)
        features = self.fusion(batch_tensor)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'graph_features': batch_tensor
        }

def create_semantic_gnn(num_classes=2, feature_dim=512):
    model = SemanticConsistencyGNN(num_classes=num_classes, feature_dim=feature_dim)
    return model 