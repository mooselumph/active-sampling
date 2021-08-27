from train import train_glow
from data import setup_data


def get_default_config():
    # Data hyperparameters for 1 GPU training
    # Some small changes to the original model so 
    # everything fits in memory
    # In particular, I had  to use shallower
    # flows (smaller K value)
    config_dict = {
        'image_path': "../../datasets/celeba/img_align_celeba",
        'train_split': 0.6,
        'image_size': 64,
        'num_channels': 3,
        'num_bits': 5,
        'batch_size': 64,
        'K': 16,
        'L': 3,
        'nn_width': 512, 
        'learn_top_prior': True,
        'sampling_temperature': 0.7,
        'init_lr': 1e-3,
        'num_epochs': 13,
        'num_warmup_epochs': 1,
        'num_sample_epochs': 0.2, # Fractional epochs for sampling because one epoch is quite long 
        'num_save_epochs': 5,
    }

    output_hw = config_dict["image_size"] // 2 ** config_dict["L"]
    output_c = config_dict["num_channels"] * 4**config_dict["L"] // 2**(config_dict["L"] - 1)
    config_dict["sampling_shape"] = (output_hw, output_hw, output_c)

    return config_dict




if __name__ == '__main__': 

    config_dict = get_default_config()

    train_ds, val_ds = setup_data(config_dict,show_grid=True)
    
    model, params = train_glow(train_ds, val_ds=val_ds, **config_dict)

