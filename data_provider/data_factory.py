from data_provider.data_loader import SMARD, SMARD_w_WEATHER #, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'smard': SMARD,
    'smard_w_weather': SMARD_w_WEATHER,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        print(args.final_run_train_on_train_and_val)
        if args.final_run_train_on_train_and_val and flag == 'val':
            shuffle_flag = False
            drop_last = False
        else:
            shuffle_flag = True
            drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        final_run=args.final_run_train_on_train_and_val
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
