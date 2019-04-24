def obs_pred_split(obs_len, pred_len, *arrays):
    obs_len_arrays = [a[:, :obs_len, ...] for a in arrays]
    pred_len_arrays = [a[:, obs_len:obs_len + pred_len, ...] for a in arrays]

    return obs_len_arrays, pred_len_arrays
