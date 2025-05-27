import pandas as pd


def make_new_txt(csv, all_txt, train_out, dev_out, eval_out):
    df = pd.read_csv(csv)
    train_list = []
    dev_list = []
    eval_list = []

    for idx, data in df.iterrows():
        print(data[0].split('/')[-1], data[-1])
        if data[-1] == 'train':
            train_list.append(data[0].split('/')[-1])
        elif data[-1] == 'dev':
            dev_list.append(data[0].split('/')[-1])
        elif data[-1] == 'eval':
            eval_list.append(data[0].split('/')[-1])

    txt_dict = {}
    cntt = 0
    with open(all_txt, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            img = line.split(' ')
            id = img[0].split('/')[0]
            if id not in txt_dict:
                txt_dict[id] = []
            txt_dict[id].append({
                "color": img[0],
                "depth": img[1],
                "ir": img[2],
                "label": img[3]
            })
            cntt += 1
            # print(line.split(' '))
    print(len(train_list) + len(dev_list) + len(eval_list), cntt)
    cnt = 0
    with open(train_out, 'w') as f:
        for id in train_list:
            for obj in txt_dict[id]:
                cnt += 1
                f.write(f"{obj['color']} {obj['depth']} {obj['ir']} {obj['label']}\n")

    with open(dev_out, 'w') as f:
        for id in dev_list:
            for obj in txt_dict[id]:
                cnt += 1
                f.write(f"{obj['color']} {obj['depth']} {obj['ir']} {obj['label']}\n")

    with open(eval_out, 'w') as f:
        for id in eval_list:
            for obj in txt_dict[id]:
                cnt += 1
                f.write(f"{obj['color']} {obj['depth']} {obj['ir']} {obj['label']}\n")

    print(cnt)

make_new_txt(
    '/home/hdd1/share/public_data/FAS-2023/WMCA/bob/PROTOCOL-grandtest.csv',
    "/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_all.txt",
    f"/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_grand_train.txt",
    f"/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_grand_dev.txt",
    f"/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_grand_eval.txt"
)

loo_base = '/home/hdd1/share/public_data/FAS-2023/WMCA/bob/PROTOCOL-LOO_'
loo_list = ['fakehead', 'flexiblemask', 'glasses', 'papermask', 'prints', 'replay', 'rigidmask']

for loo in loo_list:
    loo_path = f"{loo_base}{loo}.csv"
    make_new_txt(
        loo_path,
        "/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_all.txt",
        f"/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_LOO_{loo}_train.txt",
        f"/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_LOO_{loo}_dev.txt",
        f"/home/hdd1/share/public_data/FAS-2023/WMCA/lx/WMCA_LOO_{loo}_eval.txt"
    )
