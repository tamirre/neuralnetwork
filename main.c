#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef struct Data {
    int numberTrainingLabels;
    int numberTestLabels;
    int numberTrainingData;
    int numberTestData;

    int inputNodes;
    int outputNodes;
    int hiddenLayers;
    int hiddenLayerNodes;

    double** trainingData;
    double** testData;
    int* trainingLabels;
    int* testLabels;
} Data;

const int32_t intBigEndianToLittleEndian(int32_t bigEndianValue)
{
    return ((bigEndianValue>>24)&0xff) | // move byte 3 to byte 0
                    ((bigEndianValue<<8)&0xff0000) | // move byte 1 to byte 2
                    ((bigEndianValue>>8)&0xff00) | // move byte 2 to byte 1
                    ((bigEndianValue<<24)&0xff000000); // byte 0 to byte 3
}

void printSampleImages(Data* data)
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < data->inputNodes; j++)
        {
            if ((j+1) % 28 == 0 && j != 0) 
            {
                if (data->trainingData[i][j] > 10)
                    printf("x \n");
                else
                    printf("  \n");
            } 
            else 
            {
                if (data->trainingData[i][j] > 10)
                    printf("x ");
                else
                    printf("  ");
            }
        }
    }

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < data->inputNodes; j++)
        {
            if ((j+1) % 28 == 0 && j != 0) 
            {
                if (data->testData[i][j] > 10)
                    printf("x \n");
                else
                    printf("  \n");
            } 
            else 
            {
                if (data->testData[i][j] > 10)
                    printf("x ");
                else
                    printf("  ");
            }
        }
    }
}


void printSampleLabels(Data* data)
{
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", data->trainingLabels[i]);
        printf("\n");
    }
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", data->testLabels[i]);
        printf("\n");
    }
}

void freeData(Data* data)
{
    // Free the allocated memory
    for (int i = 0; i < data->numberTrainingData; i++) {
        free(data->trainingData[i]);
    }
    free(data->trainingData);

    for (int i = 0; i < data->numberTestData; i++) {
        free(data->testData[i]);
    }
    free(data->testData);

    free(data->testLabels);
    free(data->trainingLabels);
}

int readMNISTImageData(Data* data)
{
    FILE* fileStreamTraining;
    FILE* fileStreamTest;
    errno_t error;
    if ((error = fopen_s(&fileStreamTraining, "./mnist_train_images.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_train_images.bin");
        exit(1);
    }
    
    int32_t magicNumber, numberImages, numberRows, numberCols;
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberImages, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberRows, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberCols, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);

    const int32_t magicNumberBigEndianTraining = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberImagesBigEndianTraining = intBigEndianToLittleEndian(numberImages);
    const int32_t numberRowsBigEndianTraining = intBigEndianToLittleEndian(numberRows);
    const int32_t numberColsBigEndianTraining = intBigEndianToLittleEndian(numberCols);

    data->inputNodes = numberRowsBigEndianTraining * numberColsBigEndianTraining;
    data->numberTrainingData = numberImagesBigEndianTraining;
    
    data->trainingData = malloc(data->numberTrainingData * sizeof(double*));
    if (data->trainingData == NULL)
    {
        perror("Failed to allocate memory for trianingData!");
        return 1;
    }
    for (int i = 0; i < data->numberTrainingData; i++) {
        data->trainingData[i] = malloc(data->inputNodes * sizeof(double));
        if (data->trainingData[i] == NULL) {
            perror("Failed to allocate memory for row");
            // Free already allocated memory in case of failure
            for (int j = 0; j < i; j++) {
                free(data->trainingData[j]);
            }
            free(data->trainingData);
            return 1;
        }
    }

    for (int i = 0; i < data->numberTrainingData; i++)
    {
        for (int j = 0; j < data->inputNodes; j++)
        {
            unsigned char pixel;
            fread_s(&pixel, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTraining);
            data->trainingData[i][j] = (double) pixel;
        }
    }

    // data->trainingData = trainingData;
    if ((error = fopen_s(&fileStreamTest, "./mnist_test_images.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_test_images.bin");
        exit(1);
    }
    
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberImages, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberRows, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberCols, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);

    const int32_t magicNumberBigEndianTest = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberImagesBigEndianTest = intBigEndianToLittleEndian(numberImages);

    data->numberTestData = numberImagesBigEndianTest;
    data->testData = malloc(data->numberTestData * sizeof(double*));
    if (data->testData == NULL)
    {
        perror("Failed to allocate memory for testData!");
        return 1;
    }
    for (int i = 0; i < data->numberTestData; i++) {
        data->testData[i] = malloc(data->inputNodes * sizeof(double));
        if (data->testData[i] == NULL) {
            perror("Failed to allocate memory for row");
            // Free already allocated memory in case of failure
            for (int j = 0; j < i; j++) {
                free(data->testData[j]);
            }
            free(data->testData);
            return 1;
        }
    }

    for (int i = 0; i < data->numberTestData; i++)
    {
        for (int j = 0; j < data->inputNodes; j++)
        {
            unsigned char pixel;
            fread_s(&pixel, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTest);
            data->testData[i][j] = (double) pixel;
        }
    }
    printf("done reading images");
    // print some training samples
    printSampleImages(data);

    fclose(fileStreamTraining);
    fclose(fileStreamTest);
    return 0;
}


int readMNISTLabelData(Data* data)
{
    FILE* fileStreamTraining;
    FILE* fileStreamTest;
    errno_t error;
    if ((error = fopen_s(&fileStreamTraining, "./mnist_train_labels.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_train_labels.bin");
        exit(1);
    }
     
    int32_t magicNumber, numberLabels; 
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberLabels, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);

    const int32_t magicNumberBigEndian = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberLabelsBigEndian = intBigEndianToLittleEndian(numberLabels);
    
    data->numberTrainingLabels = numberLabelsBigEndian;
    data->trainingLabels = malloc(data->numberTrainingLabels * sizeof(int));
    if (data->trainingLabels == NULL)
    {
        perror("Failed to allocate memory for testData!");
        return 1;
    }

    // printf("%d\n", data->numberTrainingLabels);

    if ((error = fopen_s(&fileStreamTest, "./mnist_test_labels.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_test_labels.bin");
        exit(1);
    }
    
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberLabels, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);

    const int32_t magicNumberBigEndianTraining = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberLabelsBigEndianTraining = intBigEndianToLittleEndian(numberLabels);
    
    data->numberTestLabels = numberLabelsBigEndianTraining;
    data->testLabels = malloc(data->numberTestLabels * sizeof(int));
    if (data->testLabels == NULL)
    {
        perror("Failed to allocate memory for testData!");
        return 1;
    }

    for (int i = 0; i < data->numberTrainingLabels; i++)
    {
        unsigned char label;
        fread_s(&label, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTraining);
        data->trainingLabels[i] = (int)label;

    }

    for (int i = 0; i < data->numberTestLabels; i++)
    {
        unsigned char label;
        fread_s(&label, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTest);
        data->testLabels[i] = (int)label;
    }

    // print some training samples
    printSampleLabels(data);
    
    fclose(fileStreamTraining);
    fclose(fileStreamTest);
    return 0;
}

double sigmoidFunction (double x)
{
    return (1.0/(1.0+exp(-x)));
}

int main() {

    Data data = {
        .outputNodes = 10,
        .hiddenLayers = 1,
        .hiddenLayerNodes = 10,
    };
    
    if(readMNISTImageData(&data)) return 1;
    if(readMNISTLabelData(&data)) return 1;
    printf("Read data: complete!\n");
    
    freeData(&data);
    printf("Freed Data.\n");
    return 0;
}