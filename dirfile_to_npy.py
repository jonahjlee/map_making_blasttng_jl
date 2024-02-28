import numpy as np
import struct
import os


# we don't have ctime_built for these folders :(


def readFileIntoLines(file_path):
    '''Split a file into a list of lines.
    '''

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read all lines from the file and store them in a list
            lines = file.readlines()
        return lines
    
    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"File '{file_path}' not found.")
        return None
    
    except Exception as e:
        # Handle other exceptions
        print(f"Error reading file '{file_path}': {e}")
        return None


def mapTypeToStructFormat(typeStr):
    '''Numpy str format type to struct.unpack str format code.
    '''

    type_mapping = { # for struct.unpack
        'int8': 'b',
        'int16': 'h',
        'int32': 'i',
        'int64': 'q',
        'uint8': 'B',
        'uint16': 'H',
        'uint32': 'I',
        'uint64': 'Q',
        'float32': 'f',
        'float64': 'd',
        }
    return type_mapping[typeStr]


def splitDirfileLine(line):
    '''Split dirfile format file line into words.
    Some checking and conversion is done.
    '''

    words = line.split()

    # if len(words) < 4:
    #     print(f"Error: Line contains only {len(words)} words. 4 expected. Line:{line}")
    #     return False

    fname, line_type, type_str, end_num = words
    return (fname, line_type, type_str.lower(), end_num)


def typeStrToStructStr(type_str, buf_len, little_endian):
    '''Convert numpy type str to struct type str.
    '''

    # Calculate the count dynamically based on binary_data_length
    type_size = np.dtype(type_str.lower()).itemsize
    count = buf_len // type_size

    # struct.unpack format string
    struct_string = f'{count}{mapTypeToStructFormat(type_str)}'

    # endianess
    endian_char = '<' if little_endian else '>'
    struct_string = endian_char + struct_string

    return struct_string


def subDirfileToNpy(f_in, f_out, line, write, little_endian):
    '''Convert subdirfile to npy

    f_in: (str) Absolute subdirfile.
    f_out: (str) Absolute npy file to save.
    line: (str) Format file line for this file.
    write: (bool) If False, do a mock run.
    '''

    try:
        f = open(f_in, "rb")
        buf = f.read()
        _, _, type_str, _ = splitDirfileLine(line)
        struct_str = typeStrToStructStr(type_str, len(buf), little_endian)
        # print(struct_str)
        x = np.array(struct.unpack(struct_str, buf), dtype=type_str)
        if write:
            os.makedirs(os.path.split(f_out)[0], exist_ok=True)
            np.save(f_out, x)
        f.close()

        print(f" Saved as NPY: {f_out}")

        return True

    # except:
    except Exception as e:
        print(f"Exception: {e}")
        return False


# create a function to take a folder representing a dirfile and convert to a folder containing npy files in some other place
# use the FORMAT file to do it properly instead of assuming int32
def dirfile_to_npy(dname_dirfile, dname_dest, write):
    '''Convert dirfile sub file to numpy format.
    '''

    # try to open the format file (or skip this dir)
    fname_format = f"{dname_dirfile}/format"
    lines = readFileIntoLines(fname_format)
    if lines is None:
        return # error or nothing useful in format file

    # iterate through the lines in the format file
    little_endian = False
    for line in lines:

        if line.startswith('/ENDIAN little'):
            little_endian = True

        # skip blank lines, and lines starting with a hash
        if line.isspace() or line.startswith('#') or line.startswith('/'):
            continue    

        # parse the line
        try:
            fname, line_type, type_str, end_num = splitDirfileLine(line)
        except:
            continue

        # skip lines that aren't RAW (actual files)
        if line_type != 'RAW':
            continue

        # convert file to npy
        f_in = os.path.join(dname_dirfile, fname)
        f_out = os.path.join(dname_dest, fname)
        subDirfileToNpy(f_in, f_out, line, write, little_endian)


# create a function to dir walk some given directory and perform conversions in a mirrored dir structure at some given directory base
def convertDirfiles(dname_root, dname_dest_root, write=False):
    '''Walk given dir (recursively) and convert dirfiles to numpy format.
    '''

    # start walking dir
    for root, dirs, files in os.walk(dname_root):
        for dir in dirs:
            dname_dirfile = os.path.join(root, dir)
            dname_dest = os.path.join(dname_dest_root, dir)

            print(f"{dname_dirfile}... ")

            # convert this dirfile
            dirfile_to_npy(dname_dirfile, dname_dest, write)



if __name__ == '__main__':
    convertDirfiles(dname_root='/media/player1/blast2020fc1/fc1/extracted', dname_dest_root='/media/player1/blast2020fc1/fc1/converted', write=True)



    # example format file contents:
'''
# Linklist Dirfile Format File
# Auto-generated by linklist_writer

/VERSION 10
/ENDIAN little
/PROTECT none
/ENCODING none
ll_filename STRING /media/javier/blast2020fc1/fc1/extracted/roach1_2020-01-06-04-37-54
i_kid0000_roach1 RAW FLOAT32 1
...
q_kid1015_roach1 RAW FLOAT32 1
header_roach1 RAW UINT64 1
ctime_roach1 RAW UINT32 1
pps_count_roach1 RAW UINT32 1
clock_count_roach1 RAW UINT32 1
packet_count_roach1 RAW UINT32 1
status_reg_roach1 RAW UINT32 1
ll_framenum RAW UINT32 1
ll_data_integrity RAW FLOAT32 1
ll_local_time RAW UINT32 1

####### Begin calspecs ######
'''
# the lines with / in front are info about the files
    # I think we'll ignore these lines for this project
# each line represents a file in the folder (mostly)
    # filename RAW type 1
    # it looks like RAW are files, STRING is a variable
    # no idea what the ending 1 is
    # can we get a definitive list of types?





# def parseDirfileFormatLine(buf_len, line):
#     '''Name, dtype_string, and format_string from DIRFILE format file line.

#     buf_len: (int) Length of file.
#     line: (str) Format file line for file.
#     '''
    
#     # Extract relevant information using regular expression
#     match = re.match(r'(\S+)\s+RAW\s+(\S+)\s+\d+', line)
#     if match:
#         name, dirfile_type = match.groups()

#         # Calculate the count dynamically based on binary_data_length
#         type_size = np.dtype(dirfile_type.lower()).itemsize
#         count = buf_len // type_size

#         # NumPy dtype string
#         dtype_string = f'{dirfile_type.lower()}'

#         # struct.unpack format string
#         # assuming little endian
#         format_string = f'<{count}{mapTypeToStructFormat(dtype_string)}'

#         return name, dtype_string, format_string
    
#     else:
#         raise ValueError(f"Invalid format line: {line}")