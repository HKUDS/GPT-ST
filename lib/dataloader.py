import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, input_base_dim, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            scaler = StandardScaler(mean, std)
            data[:, :, 0:input_base_dim] = scaler.transform(data[:, :, 0:input_base_dim])
        else:
            data_ori = data[:, :, 0:input_base_dim]
            data_day = data[:, :, input_base_dim:input_base_dim+1]
            data_week = data[:, :, input_base_dim+1:input_base_dim+2]

            mean_data = data_ori.mean()
            std_data = data_ori.std()
            mean_day = data_day.mean()
            std_day = data_day.std()
            mean_week = data_week.mean()
            std_week = data_week.std()

            scaler_data = StandardScaler(mean_data, std_data)
            data_ori = scaler_data.transform(data_ori)
            scaler_day = StandardScaler(mean_day, std_day)
            data_day = scaler_day.transform(data_day)
            scaler_week = StandardScaler(mean_week, std_week)
            data_week = scaler_week.transform(data_week)
            data = np.concatenate([data_ori, data_day, data_week], axis=-1)
            print(mean_data, std_data, mean_day, std_day, mean_week, std_week)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler_data, scaler_day, scaler_week, None
    # return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(args, X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset, args)        # B, N, D
    #normalize st data
    # data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)

    print('============', data_train.shape, data_val.shape, data_test.shape)
    _, scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(data_train, normalizer, args.input_base_dim, args.column_wise)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    x_tra_data = scaler_data.transform(x_tra[:, :, :, :args.input_base_dim])
    y_tra_data = scaler_data.transform(y_tra[:, :, :, :args.input_base_dim])
    x_tra_day = scaler_day.transform(x_tra[:, :, :, args.input_base_dim:args.input_base_dim+1])
    y_tra_day = scaler_day.transform(y_tra[:, :, :, args.input_base_dim:args.input_base_dim+1])
    x_tra_week = scaler_week.transform(x_tra[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
    y_tra_week = scaler_week.transform(y_tra[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
    x_tra = np.concatenate([x_tra_data, x_tra_day, x_tra_week], axis=-1)
    y_tra = np.concatenate([y_tra_data, y_tra_day, y_tra_week], axis=-1)

    x_val_data = scaler_data.transform(x_val[:, :, :, :args.input_base_dim])
    y_val_data = scaler_data.transform(y_val[:, :, :, :args.input_base_dim])
    x_val_day = scaler_day.transform(x_val[:, :, :, args.input_base_dim:args.input_base_dim+1])
    y_val_day = scaler_day.transform(y_val[:, :, :, args.input_base_dim:args.input_base_dim+1])
    x_val_week = scaler_week.transform(x_val[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
    y_val_week = scaler_week.transform(y_val[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
    x_val = np.concatenate([x_val_data, x_val_day, x_val_week], axis=-1)
    y_val = np.concatenate([y_val_data, y_val_day, y_val_week], axis=-1)

    x_test_data = scaler_data.transform(x_test[:, :, :, :args.input_base_dim])
    y_test_data = scaler_data.transform(y_test[:, :, :, :args.input_base_dim])
    x_test_day = scaler_day.transform(x_test[:, :, :, args.input_base_dim:args.input_base_dim+1])
    y_test_day = scaler_day.transform(y_test[:, :, :, args.input_base_dim:args.input_base_dim+1])
    x_test_week = scaler_week.transform(x_test[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
    y_test_week = scaler_week.transform(y_test[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
    x_test = np.concatenate([x_test_data, x_test_day, x_test_week], axis=-1)
    y_test = np.concatenate([y_test_data, y_test_day, y_test_week], axis=-1)

    ##############get dataloader######################
    train_dataloader = data_loader(args, x_tra, y_tra, args.batch_size, shuffle=True, drop_last=False)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(args, x_val, y_val, args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = data_loader(args, x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler_data, scaler_day, scaler_week, scaler_holiday