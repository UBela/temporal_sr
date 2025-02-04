import torch


def get_wind_speed(u , v):
        
    return torch.sqrt(u ** 2 + v ** 2)

def denormalize(data, means):
    for i in range(data.shape[1]):
        data[:, i, :, :] += means[i]
    return data

def average_pooling(data, scale, to_int=False):

    new_h, new_w = data.shape[0] // scale, data.shape[1] // scale
    data = data.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))
    if to_int:
        data = np.uint8(data)
    return data
    
def calc_mse(pred, label):
    return torch.mean((pred - label) ** 2)

def calc_mae(pred, label):
    return torch.mean(torch.abs(pred - label))

def rescale_data(data, custom_scale = None):
    returned_scale = {}
    for var in data:
        
        if custom_scale and var in custom_scale:
            min_val = custom_scale[var]['min']
            max_val = custom_scale[var]['max']
        else:
            min_val = data[var].values.min()
            max_val = data[var].values.max()
            
            returned_scale[var] = {'min': min_val, 'max': max_val}
            
            
        # Rescale to the target range [0, 1]
        data[var] = (data[var] - min_val) / (max_val - min_val)  
        
        
    return data, returned_scale



