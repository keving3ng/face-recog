#!env/bin/python

import os

class Image_Loader():
    DIRECTORY = ""

    def __init__(self, directory):
        if not os.path.exists(directory):
            raise Exception("Invalid Directory \"{}\"".format(directory))
        else: 
            self.DIRECTORY = directory

    def getDirectory(self):
        return self.DIRECTORY

    def listDirectory(self):
        return os.listdir(self.DIRECTORY)

    def countDirectory(self):
        return len(self.listDirectory())

    def getTypes(self):
        exts = set([x[-1] for x in self.listDirectory()])
        return len(exts)