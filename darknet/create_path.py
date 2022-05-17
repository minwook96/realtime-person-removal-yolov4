import glob
def file_path_save():
    filenames = []
    files = sorted(glob.glob("./obj/*.jpg"))
    for i in range(len(files)):
        f = open("./train.txt", 'a')
        f.write(files[i] + "\n")


if __name__ == '__main__':
    file_path_save()