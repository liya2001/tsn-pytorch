import os


def build_list(data_path, save_name, flow=False):
    activity_list = []
    for act_type in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, act_type)):
            continue
        label = int(act_type[0]) - 1 if int(act_type[0]) < 8 else int(act_type[0]) - 2
        for activity in os.listdir(os.path.join(data_path, act_type)):
            if flow:
                img_range = list(map(lambda x: int(x[:-6]), os.listdir(os.path.join(data_path, act_type, activity))))
            else:
                img_range = list(map(lambda x: int(x[:-4]), os.listdir(os.path.join(data_path, act_type, activity))))
            activity_path = os.path.join(data_path.split('/')[-2], data_path.split('/')[-1], act_type, activity)
            # activity path, frames count, label, offset
            activity_list.append('{} {} {} {}'.format(activity_path,
                                                      len(img_range)//2 if flow else len(img_range),
                                                      label, min(img_range)))
    with open(save_name, 'w') as f:
        f.write('\n'.join(activity_list))


if __name__ == '__main__':
    data_path = '/home/liya/workspace/trecvid/data/candidate_region'
    save_path = '/home/liya/workspace/trecvid/tsn-pytorch/data'

    train_rgb = os.path.join(data_path, 'train/gt_activities')
    build_list(train_rgb, os.path.join(save_path, 'train_rgb.txt'))

    train_flow = os.path.join(data_path, 'train/gt_acts_opt_flow')
    build_list(train_flow, os.path.join(save_path, 'train_flow.txt'), flow=True)

    val_rgb = os.path.join(data_path, 'val/gt_activities')
    build_list(val_rgb, os.path.join(save_path, 'val_rgb.txt'))

    val_flow = os.path.join(data_path, 'val/gt_acts_opt_flow')
    build_list(val_flow, os.path.join(save_path, 'val_flow.txt'), flow=True)


