# -*- coding: ISO-8859-1 -*-

# standard library imports
import codecs
import sys
# from types import StringType
from io import StringIO, BytesIO
from struct import unpack
# from cStringIO import StringIO

# custom import
from numpy import unicode

from DataTypeConverters import writeBew, writeVar, fromBytes


def to_bytes(s):
    if type(s) is bytes:
        return s
    elif type(s) is str or (sys.version_info[0] < 3 and type(s) is unicode):
        return codecs.encode(s, 'ascii')
    else:
        raise TypeError("Expected bytes or string, but got %s." % type(s))


class RawOutstreamFile:
    """
    
    Writes a midi file to disk.
    
    """

    def __init__(self, outfile=''):
        self.buffer = BytesIO()
        self.outfile = outfile

    # native data reading functions

    def writeSlice(self, str_slice):
        """Writes the next text slice to the raw data"""
        print(str_slice)
        self.buffer.write(to_bytes(str_slice))

    def writeBew(self, value, length=1):
        """Writes a value to the file as big endian word"""
        self.writeSlice(writeBew(value, length))

    def writeVarLen(self, value):
        "Writes a variable length word to the file"
        var = self.writeSlice(writeVar(value))

    def write(self):
        "Writes to disc"
        if self.outfile:
            if isinstance(self.outfile, str):
                outfile = open(self.outfile, 'wb')
                print('hobo hobo')
                print(self.getvalue())
                outfile.write(self.getvalue())
                outfile.close()
            else:
                self.outfile.write(self.getvalue())
        else:
            sys.stdout.write(self.getvalue().decode())

    def getvalue(self):
        return self.buffer.getvalue()


if __name__ == '__main__':
    out_file = 'test/midifiles/midiout.mid'
    out_file = ''
    rawOut = RawOutstreamFile(out_file)
    rawOut.writeSlice(str.encode('MThd'))
    rawOut.writeBew(6, 4)
    rawOut.writeBew(1, 2)
    rawOut.writeBew(2, 2)
    rawOut.writeBew(15360, 2)
    rawOut.write()
