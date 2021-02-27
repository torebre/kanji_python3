import struct
import numpy as np
from matplotlib import pylab as plt

from skimage.morphology import skeletonize
from skimage.transform import resize

structFormat = 'HH4s504s64s'
structLength = struct.calcsize(structFormat)
unpackFunction = struct.Struct(structFormat).unpack_from

kanjiData = []

with open('/home/student/Downloads/ETL/ETL9B/ETL9B_2', mode='rb') as file:
    # The first record is a dummy record
    record = file.read(structLength)

    # for i in range(1, 10):
    while True:
        record = file.read(structLength)

        if not record:
            break

        (serialSheetNumber, kanjiCode, typicalReading, imageData, uncertain) = unpackFunction(record)
        image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape((63, 64))

        kanjiData.append(tuple([serialSheetNumber, kanjiCode, typicalReading, image]))

seenKanjiCodes = {}
counter = 0

for record in kanjiData:
    # skeletonizedImage = skeletonize(record[3])
    # resizedImage = resize(skeletonizedImage, (32, 32))
    # plt.imsave('../etl_sample_pictures2/preprocessed_kanji_' + str(record[0]) + '_' + str(record[1]) + '.png', resizedImage)

    if record[0] == 9219 and record[1] == 12068:
        b = '\n'.join('\t'.join('%d' % x for x in y) for y in record[3])
        print "Image: ", b
        print "Sum: ", record[3].sum()

    plt.imsave('../etl_sample_pictures2/preprocessed_kanji_' + str(record[0]) + '_' + str(record[1]) + '.png',
               record[3])

    # print "Kanji code: ", str(record[1])

    count = seenKanjiCodes.get(record[1], 0)

    seenKanjiCodes[record[1]] = count + 1

    if counter == 10:
        break

    counter += 1

# print seenKanjiCodes

print "Number of distinct kanji codes: ", len(seenKanjiCodes)
