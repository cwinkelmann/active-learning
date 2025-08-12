def get_dataset_image_merged_filesname(dsn, imn, iid):
    return f"{imn}"
    # return f"{dsn}::{iid}::{imn}"

def get_dataset_image_merged_filesname_v2(dsn, imn):
    return f"{dsn}___{imn}"
    # return f"{dsn}::{iid}::{imn}"


def get_image_name(iid, suffix):
    return f"{iid}.{suffix}"