import struct
import numpy as np

STRUCT_FORMAT = 'hh4s504s64s'
ETL_BASE_PATH = ['/home/student/Downloads/ETL/ETL9B/ETL9B_']


def readall():
    structLength = struct.calcsize(STRUCT_FORMAT)
    unpack_function = struct.Struct(STRUCT_FORMAT).unpack_from

    kanjiData = []

    for i in range(1, 6):
        with open('/home/student/Downloads/ETL/ETL9B/ETL9B_' +str(i), mode = 'rb') as file:
            # The first record is a dummy record
            file.read(structLength)

            while True:
                record = file.read(structLength)

                if not record:
                    break

                (serialSheetNumber, kanjiCode, typicalReading, imageData, uncertain) = unpack_function(record)
                image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape((63, 64))

                kanjiData.append({'serialSheetNumber': serialSheetNumber, 'kanjiCode': kanjiCode, 'typicalReading': typicalReading, 'image': image})

    return kanjiData