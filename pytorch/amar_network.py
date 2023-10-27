from __future__ import annotations

import torch
import torch.nn as nn


class AMARNetwork(torch.nn.Module):

    def __init__(self, device, grad_features=False, custom_weights=None):

        super().__init__()

        self.device = device
        self.grad_features = grad_features
        self.custom_weights = custom_weights

    def init_(self):
        if self.custom_weights is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            i = 0
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight = torch.nn.Parameter(torch.from_numpy(self.custom_weights[i]))
                    i += 1
                    nn.init.zeros_(m.bias)

        self.to(self.device)

    def return_scores(self, user_idx, item_idx):
        with torch.no_grad():
            scores = self((torch.from_numpy(user_idx).to(self.device).long(),
                           torch.from_numpy(item_idx).to(self.device).long())).cpu()

            if len(item_idx) != 1:
                return scores.squeeze()
            else:
                return scores[0]


class SingleSourceAMARNetwork(AMARNetwork):

    def __init__(self, items_features, users_features, device, grad_features=False, custom_weights=None):

        super().__init__(device, grad_features, custom_weights)

        if len(users_features) != 1:
            raise ValueError("This class only supports a single source of features\n"
                             f"The following number of users features were found: {len(users_features)}")

        if len(items_features) != 1:
            raise ValueError("This class only supports a single source of features\n"
                             f"The following number of items features were found: {len(items_features)}")

        self.users_features = torch.nn.Parameter(users_features[0].float().to(self.device), requires_grad=self.grad_features)
        self.items_features = torch.nn.Parameter(items_features[0].float().to(self.device), requires_grad=self.grad_features)


class AMARNetworkBasic(SingleSourceAMARNetwork):

    def __init__(self, items_features, users_features, device, grad_features=False, custom_weights=None):

        super().__init__(items_features, users_features, device, grad_features, custom_weights)

        self.dense_user = torch.nn.Sequential(
            nn.Linear(self.users_features.size(1), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.dense_item = torch.nn.Sequential(
            nn.Linear(self.items_features.size(1), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.compute_score = torch.nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.init_()

    def forward(self, x):
        user_idx = x[0].to(self.device)
        item_idx = x[1].to(self.device)

        users_features = self.users_features[user_idx]
        items_features = self.items_features[item_idx]

        dense_users = self.dense_user(users_features)
        dense_items = self.dense_item(items_features)

        out = self.compute_score(torch.cat([dense_users, dense_items], dim=-1))

        return out


class DoubleSourceAMARNetwork(AMARNetwork):

    def __init__(self, items_features: torch.Tensor, users_features: torch.Tensor, device, grad_features=False, custom_weights=None):

        super().__init__(device, grad_features, custom_weights)

        if len(users_features) != 2:
            raise ValueError("This class only supports two sources of features\n"
                             f"The following number of users features were found: {len(users_features)}")

        if len(items_features) != 2:
            raise ValueError("This class only supports two sources of features\n"
                             f"The following number of items features were found: {len(items_features)}")

        self.first_users_features = torch.nn.Parameter(users_features[0].float().to(self.device), requires_grad=self.grad_features)
        self.first_items_features = torch.nn.Parameter(items_features[0].float().to(self.device), requires_grad=self.grad_features)
        self.second_users_features = torch.nn.Parameter(users_features[1].float().to(self.device), requires_grad=self.grad_features)
        self.second_items_features = torch.nn.Parameter(items_features[1].float().to(self.device), requires_grad=self.grad_features)


class AMARNetworkEntityBasedConcat(DoubleSourceAMARNetwork):

    def __init__(self, items_features, users_features, device, grad_features=False, custom_weights=None):
        super().__init__(items_features, users_features, device, grad_features, custom_weights)

        self.first_dense_user = torch.nn.Sequential(
            nn.Linear(self.first_users_features.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.first_dense_item = torch.nn.Sequential(
            nn.Linear(self.first_items_features.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.second_dense_user = torch.nn.Sequential(
            nn.Linear(self.second_users_features.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.second_dense_item = torch.nn.Sequential(
            nn.Linear(self.second_items_features.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.linear_user = torch.nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.linear_item = torch.nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.compute_score = torch.nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.init_()

    def forward(self, x):
        user_idx = x[0].to(self.device)
        item_idx = x[1].to(self.device)

        first_users_features = self.first_users_features[user_idx]
        first_items_features = self.first_items_features[item_idx]
        second_users_features = self.second_users_features[user_idx]
        second_items_features = self.second_items_features[item_idx]

        x1_user = self.first_dense_user(first_users_features)
        x1_item = self.first_dense_item(first_items_features)

        x2_user = self.second_dense_user(second_users_features)
        x2_item = self.second_dense_item(second_items_features)

        concat_user = self.linear_user(torch.cat([x1_user, x2_user], dim=-1))
        concat_item = self.linear_item(torch.cat([x1_item, x2_item], dim=-1))
        out = self.compute_score(torch.cat([concat_user, concat_item], dim=-1))

        return out


class AMARNetworkMerge(DoubleSourceAMARNetwork):

    def __init__(self, items_features, users_features, device, grad_features=False, custom_weights=None, drop_value=0.4):
        super().__init__(items_features, users_features, device, grad_features, custom_weights)

        self.first_dense_user = torch.nn.Sequential(
            nn.Dropout(drop_value),
            nn.Linear(self.first_users_features.size(1), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.first_dense_item = torch.nn.Sequential(
            nn.Dropout(drop_value),
            nn.Linear(self.first_items_features.size(1), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.second_dense_user = torch.nn.Sequential(
            nn.Dropout(drop_value),
            nn.Linear(self.second_users_features.size(1), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.second_dense_item = torch.nn.Sequential(
            nn.Dropout(drop_value),
            nn.Linear(self.second_items_features.size(1), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.attention_user = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.Softmax(dim=-1),
        )

        self.attention_item = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.Softmax(dim=-1),
        )

        self.attention_cross = torch.nn.Sequential(
            nn.Linear(1, 128),
            nn.Softmax(dim=-1),
        )

        self.compute_score = torch.nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.init_()

    @staticmethod
    def merge(x_1, x_2, attention):
        return attention * x_1 + (1 - attention) * x_2

    def forward(self, x):
        user_idx = x[0].to(self.device)
        item_idx = x[1].to(self.device)

        first_users_features = self.first_users_features[user_idx]
        first_items_features = self.first_items_features[item_idx]
        second_users_features = self.second_users_features[user_idx]
        second_items_features = self.second_items_features[item_idx]

        x1_user = self.first_dense_user(first_users_features)
        x1_item = self.first_dense_item(first_items_features)

        x2_user = self.second_dense_user(second_users_features)
        x2_item = self.second_dense_item(second_items_features)

        concat_user = torch.cat([x1_user, x2_user], dim=-1)
        attention_user = self.attention_user(concat_user)
        merged_user = self.merge(x1_user, x2_user, attention_user)

        concat_item = torch.cat([x1_item, x2_item], dim=-1)
        attention_item = self.attention_item(concat_item)
        merged_item = self.merge(x1_item, x2_item, attention_item)

        attention_weights = self.attention_cross(torch.sum(merged_user * merged_item, dim=-1).unsqueeze(-1))
        merged_item_user = torch.add(merged_user * attention_weights, merged_item * (1 - attention_weights))

        out = self.compute_score(merged_item_user)

        return out
