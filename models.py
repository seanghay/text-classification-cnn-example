import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=[100, 100, 100],
        num_classes=10,
        dropout=0.5,
    ):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=3,
            max_norm=0.5,
        )
        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters[i],
                    kernel_size=filter_sizes[i],
                )
                for i in range(len(filter_sizes))
            ]
        )
        self.fc = nn.Linear(sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        x_embed = self.embedding(input_ids).float()
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [
            F.max_pool1d(
                x_conv,
                kernel_size=x_conv.shape[2]
                if isinstance(x_conv.shape[2], int)
                else x_conv.shape[2].item(),
            )
            for x_conv in x_conv_list
        ]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits


if __name__ == "__main__":
    model = TextCNN()
    batch_size = 1  # Example batch size
    max_seq_len = 128  # Example maximum sequence length
    input_ids = torch.randint(low=0, high=8000, size=(batch_size, max_seq_len))
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=1).flatten()
        print(preds)
