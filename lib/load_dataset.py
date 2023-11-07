import os
import numpy as np

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    # index_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data


def load_st_dataset(dataset, args):
    if dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    elif dataset == 'METR_LA':
        data_path = os.path.join('../data/METR_LA/metr_la.npz')
        data = np.load(data_path)['data']  # only traffic speed data
        print(data.shape, data[data == 0].shape)
        # print(sss)
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        week_start = 4
        holiday_list = [88]
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False,
                                                     holiday_list=holiday_list)
    elif dataset == 'NYC_BIKE':
        data_path = os.path.join('../data/NYC_BIKE/NYC_BIKE.npz')
        data = np.load(data_path)['data']  # DROP & PICK
        week_start = 5
        weekday_only = False
        interval = 30
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data[..., 0], week_start, interval, weekday_only, holiday_list=holiday_list)
    elif dataset == 'NYC_TAXI':
        data_path = os.path.join('../data/NYC_TAXI/NYC_TAXI.npz')
        data = np.load(data_path)['data']  # DROP & PICK
        week_start = 5
        weekday_only = False
        interval = 30
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data[..., 0], week_start, interval, weekday_only, holiday_list=holiday_list)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    elif len(data.shape) > 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data
