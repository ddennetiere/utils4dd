# -*- coding: utf-8 -*-

## @file ZygoDatx.py
# This file provides tools for analysing surface height data with statistical and PSD tools.
#
# The file defines the HeightMap class, which is the main storage element of surface height measurements.
#   @author François Polack, Uwe Flechsig
#   @copyright 2022, Superflat-PCP project  <mailto:superflat-pcp@synchrotron-soleil.fr>
#   @version 0.1.0


import tkinter
from tkinter import filedialog
from pathlib import Path
import h5py as h5
import numpy as np

class Node:


    def __init__(self, name, value=None):
        self.name=name
        self.value=value
        self.parent=None
        self.children=[]

    def append_node(self,node):
        if not isinstance(node, Node):
            raise TypeError("Argument of append_node mus be an instance of class Node")
        self.children.append(node)
        node.parent=self
        return node

    def __getitem__(self, key):
        for node in self.children:
            if node.name==key:
                return node
        raise KeyError

    def __iter__(self):
        return self.children.__iter__()

    def __len__(self):
        return len(self.children)

    def __setitem__(self, key, value):
        for nnd in self.children:
            if nnd.name==key:
                self.children.remove(nnd)
        if not isinstance(key,type("")):
            raise TypeError("Key must be of 'str' type")
        node=Node(key,value)
        self.children.append(node)
        node.parent=self

    def dump(self, level=0):
        indent="\t" * level
        for node in self.children:
            print(indent, node.name,':' , node.value)
            if len(node)!=0:
                node.dump(level+1)

class DatxFile(object):
    def __init__(self):
        ## (str) The full path to access the file
        self.filepath=""
        ## (Node) The tree of metadata constructed from the 'MetaData' group of the datx file
        self.metatree=Node('Root')
        ## (int) the layout version of the file (0 if not defined)
        self.fileversion=0

        self.attributes=None


    def openFile(self, filename=''):

        if Path(filename).is_file():
            self.filepath=filename
        else:
            tkinter.Tk().withdraw()
            self.filepath=filedialog.askopenfilename(title="open a Zygo.dat file", filetypes=(("Zygo datx","*.datx")) )
        print("Opening:", self.filepath)


        h5file=h5.File(self.filepath, 'r')

        if 'File Layout Version' in h5file['Attributes'].attrs :
            self.fileversion=h5file['Attributes'].attrs['File Layout Version'][0]
        else:
            self.fileversion=0

        print("File version=", self.fileversion)

        nodedict=dict(Root=self.metatree)

        metadata=h5file["/MetaData"] # open the MetaData DS to get the link to the Surface DS

        for line in metadata:
            src=nodedict[line['Source'].decode('ascii')]
            dest=line['Destination'].decode('ascii')
            link= line['Link'].decode('ascii')
            src[link]=dest
            if dest[0]=='{':  # this is a group node we need to add it to the node list
                nodedict[dest]=src[link]

        self.attributes=h5file[self.metatree['Measurement']['Attributes'].value ].attrs
        self.info()
        return True

    def getHeight(self, multiplier=1.):
        h5file=h5.File(self.filepath, 'r')
        dsSurf=h5file[self.metatree['Measurement']['Surface']['Path'].value]
        scale_factor=multiplier*dsSurf.attrs['Z Converter'][0][2][1]*dsSurf.attrs['Z Converter'][0][2][2]*dsSurf.attrs['Z Converter'][0][2][3]


       # print("shape="+str(dsSurf.shape)+ " type="+ str(dsSurf.dtype)+ " compression="+ dsSurf.compression+ " chunks="+ str(dsSurf.chunks))

        try:
            Z=dsSurf[()]
        except OSError as e:
            if dsSurf.compression == 'szip' :
                msg="The Surface dataset of this file was compressed with szip and cannot be read by this version of h5py.\n"
                msg+="You may consider to update the hdf5.dll and hdf5_hl.dll library files located in the Lib/site-packages/h5py directory of your Conda environment"
            else:
                msg="The Surface dataset cannot be read by this version of h5py.\n"
                msg+="shape="+str(dsSurf.shape)+ " type="+ str(dsSurf.dtype)+ " compression="+ dsSurf.compression+ " chunks="+ str(dsSurf.chunks)
            raise OSError(msg)
        Z[np.greater_equal(Z, dsSurf.attrs['No Data'][0]) ]=np.nan  #  invalid data set to NaN
        return scale_factor*Z


    def getPixel(self):
        h5file=h5.File(self.filepath, 'r')
        #  converters are not formatted the same by MX and MetroPro
        # dsSurf=h5file[self.metatree['Measurement']['Surface']['Path'].value]
        # return [dsSurf.attrs['X Converter'][0][2][1], dsSurf.attrs['Y Converter'][0][2][1] ]

        if 'Data Context.Lateral Resolution:Value' in self.attributes:
            return self.attributes['Data Context.Lateral Resolution:Value'][0] 
        else:
            raise OSError("DATX format Error:  cannot find lateral resolution info")



    def getOrigin(self):
        h5file=h5.File(self.filepath, 'r')
        dsSurf=h5file[self.metatree['Measurement']['Surface']['Path'].value]
        return [dsSurf.attrs['Coordinates'][0][0],dsSurf.attrs['Coordinates'][0][1]]


    def info(self):

        softlist=('Mx','MetroPro', 'MetroBASIC', 'd2bug')
        print("Content of file", self.filepath)
        self.metatree.dump(1)

        if 'Data Context.Data Attributes.Software Info Name' in self.attributes:
            print("Acquisition software", self.attributes['Data Context.Data Attributes.Software Info Name'][0], "version",
                   self.attributes['Data Context.Data Attributes.Software Info Version'][0] )


