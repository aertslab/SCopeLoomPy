import zlib
import base64
import json
import numpy as np


def compress_encode(value):
    '''
    Compress using ZLIB algorithm and encode the given value in base64.
    '''
    return base64.b64encode(zlib.compress(json.dumps(value).encode('ascii'))).decode('ascii')

def df_to_named_matrix(df):
    arr_ip = [tuple(i) for i in df.as_matrix()]
    dtyp = np.dtype(list(zip(df.dtypes.index, df.dtypes)))
    arr = np.array(arr_ip, dtype=dtyp)
    return arr