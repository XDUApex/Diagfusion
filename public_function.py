import os
import pickle
import argparse
import yaml


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gaia_config.yaml')  # 添加默认值
    args = parser.parse_args()
    
    config_path = os.path.join('./config', args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def min_max_normalized(feature):
    feature_copy = feature.copy().astype(float)
    for i in range(len(feature_copy)):
        min_f, max_f = min(feature_copy[i]), max(feature_copy[i])
        if min_f == max_f:
            feature_copy[i] = [0]*len(feature_copy[i])
        else:
            feature_copy[i] = (feature_copy[i] - min_f) / (max_f - min_f)
    return feature_copy


def deal_config(config, key):
    new_config = {}
    for k in config[key].keys():
        if 'path' in k or 'dir' in k:
            if config[key][k] or config[key][k] == '':
                path = os.path.join(config['base_path'], config['demo_path'],
                                    config['label'], config[key][k])
                if 'dir' in k:
                    if not os.path.exists(path):
                        os.makedirs(path)
                new_config[k] = path
            else:
                new_config[k] = config[key][k]
        else:
            new_config[k] = config[key][k]

    return new_config


if __name__ == '__main__':
    config = get_config()
    print(config['fasttext']['vector_dim'])
    cur_path = os.getcwd()
    print(cur_path[:cur_path.find('unirca')])

