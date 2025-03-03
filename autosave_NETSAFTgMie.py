import pyperclip
import os
import shutil
from datetime import datetime


def autosave():
    # Copy current NET-SAFTgMie_master.py,
    # then rename to  NET-SAFTgMie_YYMMDD_HHMM.py
    now = datetime.now()  # current time
    time_str = now.strftime("%y%m%d_%H%M")  #YYMMDD_HHMM
    # print(time_str)

    # Save .py file to clipboard
    path = os.path.dirname(__file__)
    src_dir = path + "/NET_SAFTgMie_master.py"
    # print(src_dir)
    new_fname = f"NET_SAFTgMie_{time_str}.py"
    dst_dir = path + f"/_archived/{new_fname}"
    # print(dst_dir)
    shutil.copy(src_dir, dst_dir)

    # Save file name to clipboard
    pyperclip.copy(new_fname)
    print("New saved file name: ", new_fname)
    print("")


if __name__ == "__main__":
    autosave()
