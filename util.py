def get_data(batch_memory, n_features):
    states = batch_memory[:, :n_features]
    states_ = batch_memory[:, -n_features:]
    actions = batch_memory[:, n_features].astype(int)
    rewards = batch_memory[:, n_features + 1]
    return states_, states, actions, rewards
