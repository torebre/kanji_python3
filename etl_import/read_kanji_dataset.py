from etl_import.EtlData import EtlData

datasets = [ EtlData("ETL1", )]


class ReadKanjiDataset:



    def readDataset(self, etl_data: EtlData):
        etl_data.path


        with open('/home/student/Downloads/ETL/ETL8B/ETL8B2C1', mode = 'rb') as file:
            while True:
                record = file.read(structLength)
                if not record:
                    break

                (serialSheetNumber, kanjiCode, typicalReading, imageData) = unpackFunction(record)
                image = np.unpackbits(np.fromstring(imageData, dtype=np.uint8)).reshape(64, 63)

                images.append(Image.frombuffer('1', (64, 63), imageData, 'raw'))

                kanjiData.append(tuple([serialSheetNumber, kanjiCode, image]))


