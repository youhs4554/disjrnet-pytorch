from data_loader.data_loaders import VideoDataLoader


def prepare_dataset(data_dir, batch_size, sample_length, num_workers=8, fold=1, validation=True):
    """prepare dataloaders (train/val/test) for given fold

    Args:
        data_dir (str): dataset directory path.
        fold (int, optional): fold index in [1, args.n_fold]. Defaults to 1.
        validation (bool, optional): if True, return validation dataloader. Defaults to True.

    Returns:
        dataloaders (tuple): tuple of pytorch dataloaders
    """

    # init dataset
    train_loader = VideoDataLoader(
        data_dir, batch_size=batch_size, fold=fold,
        sample_length=sample_length,
        validation_split=0.05, num_workers=num_workers)
    valid_loader = train_loader.split_validation()

    test_loader = VideoDataLoader(
        data_dir, batch_size=batch_size, fold=fold, training=False, validation_split=0.0,
        sample_length=sample_length,
        num_workers=num_workers
    )

    if validation:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader
