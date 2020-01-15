import zipfile
import os
from shutil import rmtree


def prep_files(root_path, extr_path, folder_path, label):
    new_extr_path = os.path.join(root_path,"extr\\"+ label)
    os.makedirs(new_extr_path)
    with zipfile.ZipFile(extr_path,'r') as zip_ref:
        zip_ref.extractall(new_extr_path)
    if folder_path is not None:
        actual_path = os.path.join(new_extr_path, folder_path)
        for filename in os.listdir(actual_path):
            os.rename(os.path.join(actual_path, filename), os.path.join(new_extr_path, filename))

    folders = [f.path for f in os.scandir(new_extr_path) if f.is_dir()]
    for i in folders:
        rmtree(i, ignore_errors=True)

    return new_extr_path

