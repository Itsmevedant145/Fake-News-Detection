import os

# List files/folders to delete
files_to_delete = [
    '.ipynb_checkpoints',
    'my_Text.txt',
    'Untitled.ipynb',
    'Gen_News2.txt',
    'Gen_News3.txt',
    'Gen_News4.txt'
]

for item in files_to_delete:
    if os.path.isdir(item):
        try:
            os.rmdir(item)
            print(f"Deleted directory: {item}")
        except OSError:
            print(f"Directory {item} not empty, skipping or delete manually.")
    elif os.path.isfile(item):
        os.remove(item)
        print(f"Deleted file: {item}")
    else:
        print(f"{item} not found, skipping.")
