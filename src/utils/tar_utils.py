import os
import tarfile


def untar_file(file, output):
    os.makedirs(output, exist_ok=True)
    print('[dataset] Extracting tar file {file} to {path}'.format(file=file,
                                                                  path=output))
    cwd = os.getcwd()
    tar = tarfile.open(file, "r")
    os.chdir(output)
    tar.extractall()
    tar.close()
    os.chdir(cwd)
    print('[dataset] Done!')
