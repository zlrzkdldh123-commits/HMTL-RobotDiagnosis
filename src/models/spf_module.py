"""
Severity Pattern Fusion (SPF) Module
Encodes degradation characteristics across severity stages
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeverityEmbedding(nn.Module):
    """Learnable severity level embeddings"""
    def __init__(self, num_severity_levels=3, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_severity_levels, embedding_dim)
    
    def forward(self, severity_level):
        return self.embedding(severity_level)


class DomainSpecificFeatureExtractor(nn.Module):
    """Extract domain-specific features for tension and wear"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.fc(x)


class SeverityPatternFusion(nn.Module):
    """
    SPF Module: Structurally integrates severity-dependent representations
    across tension and wear domains
    
    Args:
        input_dim: dimension of input features (typically hidden_dim)
        hidden_dim: internal hidden dimension
    """
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Severity embeddings (Light=0, Medium=1, Severe=2)
        self.severity_embedding = SeverityEmbedding(num_severity_levels=3, 
                                                    embedding_dim=hidden_dim//4)
        
        # Domain-specific feature extractors
        self.tension_extractor = DomainSpecificFeatureExtractor(input_dim, hidden_dim)
        self.wear_extractor = DomainSpecificFeatureExtractor(input_dim, hidden_dim)
        
        # Severity-aware feature processors
        self.tension_severity_processor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.wear_severity_processor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Continuous severity score predictor (auxiliary)
        self.severity_score_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h):
        """
        Args:
            h: (batch_size, input_dim) - latent features from CNN backbone
        
        Returns:
            integrated_knowledge: (batch_size, hidden_dim) - fused representation
            tension_features: (batch_size, hidden_dim) - tension domain features
            wear_features: (batch_size, hidden_dim) - wear domain features
            severity_scores: (batch_size, 1) - continuous severity estimation
        """
        batch_size = h.size(0)
        device = h.device
        
        # Extract domain-specific features
        tension_feat = self.tension_extractor(h)      # (batch, hidden_dim)
        wear_feat = self.wear_extractor(h)             # (batch, hidden_dim)
        
        # Process with all severity levels
        tension_severity_features = []
        wear_severity_features = []
        
        for severity_level in range(3):  # 0: Light, 1: Medium, 2: Severe
            severity_idx = torch.tensor([severity_level] * batch_size, device=device)
            severity_emb = self.severity_embedding(severity_idx)  # (batch, embed_dim)
            
            # Tension + severity
            tension_combined = torch.cat([tension_feat, severity_emb], dim=-1)
            tension_sev = self.tension_severity_processor(tension_combined)
            tension_severity_features.append(tension_sev)
            
            # Wear + severity
            wear_combined = torch.cat([wear_feat, severity_emb], dim=-1)
            wear_sev = self.wear_severity_processor(wear_combined)
            wear_severity_features.append(wear_sev)
        
        # Fuse severity representations (using mean over severity levels)
        tension_fused = torch.mean(torch.stack(tension_severity_features), dim=0)
        wear_fused = torch.mean(torch.stack(wear_severity_features), dim=0)
        
        # Final integration
        combined = torch.cat([tension_fused, wear_fused], dim=-1)
        integrated_knowledge = self.fusion_network(combined)
        
        # Auxiliary: continuous severity score
        severity_scores = self.severity_score_predictor(integrated_knowledge)
        
        return {
            'integrated_knowledge': integrated_knowledge,
            'tension_features': tension_fused,
            'wear_features': wear_fused,
            'severity_scores': severity_scores,
            'tension_all_severities': tension_severity_features,
            'wear_all_severities': wear_severity_features
        }
