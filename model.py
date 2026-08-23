import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictModel(nn.Module):
    def __init__(self, num_layers, d_model, dff, num_heads, vocab_size):
        super(PredictModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=0.5,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ========= FEATURE PROJECTION =========
        self.fc_payload = nn.Linear(167, d_model)
        self.fc_linker = nn.Linear(167, d_model)

        self.fc_aac1 = nn.Linear(20, d_model)
        self.fc_aac2 = nn.Linear(20, d_model)
        self.fc_aac3 = nn.Linear(20, d_model)

        self.fc_t1 = nn.Linear(2592, d_model)
        self.fc_t2 = nn.Linear(1280, d_model)
        self.fc_t3 = nn.Linear(1280, d_model)

        self.fc_dar = nn.Linear(1, d_model)

        # ========= ATTENTION =========
        self.attention = nn.Linear(4096, 1)

        # ========= FINAL CLASSIFIER =========
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(4096, d_model)
        self.fc2 = nn.Linear(d_model, d_model // 2)
        self.out = nn.Linear(d_model // 2, 1)

    def forward(self, x1, x1_maccs, x2, x2_maccs,
                t1, t2, t3, aac1, aac2, aac3, t4,
                mask1=None, adjoin_matrix1=None,
                mask2=None, adjoin_matrix2=None):
        

        # ========= SEQUENCE EMBEDDING =========
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = torch.mean(x1, dim=1)
        x2 = torch.mean(x2, dim=1)

        # ========= FEATURE TRANSFORM =========
        x1_maccs = self.fc_payload(x1_maccs)
        x2_maccs = self.fc_linker(x2_maccs)

        aac1 = self.fc_aac1(aac1)
        aac2 = self.fc_aac2(aac2)
        aac3 = self.fc_aac3(aac3)

        t1 = self.fc_t1(t1)
        t2 = self.fc_t2(t2)
        t3 = self.fc_t3(t3)

        t4 = self.fc_dar(t4.unsqueeze(1))

        # ========= NORMALIZATION =========
        x1 = F.layer_norm(x1, x1.shape[1:])
        x2 = F.layer_norm(x2, x2.shape[1:])
        t1 = F.layer_norm(t1, t1.shape[1:])
        t2 = F.layer_norm(t2, t2.shape[1:])
        t3 = F.layer_norm(t3, t3.shape[1:])

        # ========= FEATURE CONCAT =========
        features = torch.cat([
            x1, x2,
            x1_maccs, x2_maccs,
            aac1, aac2, aac3,
            t1
        ], dim=1)

        # ========= ATTENTION =========
        attn_weights = torch.softmax(self.attention(features), dim=1)
        features = features * attn_weights

        # ========= CLASSIFIER =========
        x = self.dropout(features)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        output = self.out(x)

        return output