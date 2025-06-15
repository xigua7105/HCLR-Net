from ._register import TV_TRANSFORMS

@ TV_TRANSFORMS.register_module
def define_transforms(kwargs: list):
    transforms_list = []
    for t in kwargs:
        t = {k: v for k, v in t.items()}
        t_type = t.pop('type')
        transforms_list.append(TV_TRANSFORMS.get_module(t_type)(**t))
    return TV_TRANSFORMS.get_module("Compose")(transforms_list)


@ TV_TRANSFORMS.register_module
def cifar100_train_trans(**kwargs):
    cfg_train = [
        dict(type="RandomCrop", size=32, padding=4),
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomRotation", degrees=15),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404), inplace=True)
    ]
    return TV_TRANSFORMS.get_module("define_transforms")(cfg_train)


@ TV_TRANSFORMS.register_module
def cifar100_test_trans(**kwargs):
    cfg_test = [
        dict(type="ToTensor"),
        dict(type="Normalize", mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404), inplace=True)
    ]
    return TV_TRANSFORMS.get_module("define_transforms")(cfg_test)
