#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define HIDDEN_LAYERS 1
#define HIDDEN_LAYERS_NODES 10
#define INPUT_NODES 28*28 // number of pixels (28*28) in one image
#define OUTPUT_NODES 10 // digits 0-9

#define NUMBER_TRAINING_DATA 60000
#define NUMBER_TEST_DATA 10000

double trainingData[NUMBER_TRAINING_DATA][INPUT_NODES];
int trainingLabels[NUMBER_TRAINING_DATA];
double testData[NUMBER_TEST_DATA][INPUT_NODES];
int testLabels[NUMBER_TEST_DATA];

void printSampleImages()
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            if ((j+1) % 28 == 0 && j != 0) 
            {
                if (trainingData[i][j] > 10)
                    printf("x \n");
                else
                    printf("  \n");
            } 
            else 
            {
                if (trainingData[i][j] > 10)
                    printf("x ");
                else
                    printf("  ");
            }
        }
    }

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            if ((j+1) % 28 == 0 && j != 0) 
            {
                if (testData[i][j] > 10)
                    printf("x \n");
                else
                    printf("  \n");
            } 
            else 
            {
                if (testData[i][j] > 10)
                    printf("x ");
                else
                    printf("  ");
            }
        }
    }
}


void printSampleLabels()
{
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", trainingLabels[i]);
        printf("\n");
    }

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", testLabels[i]);
        printf("\n");
    }


}


void readMNISTImageData()
{
    FILE* fileStreamTraining;
    FILE* fileStreamTest;
    errno_t error;
    if ((error = fopen_s(&fileStreamTraining, "./mnist_train_images.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_train_images.bin");
        exit(1);
    }
    
    int32_t magicNumber;
    int32_t numberImages; 
    int32_t numberRows;
    int32_t numberCols;
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberImages, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberRows, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberCols, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);

    if ((error = fopen_s(&fileStreamTest, "./mnist_test_images.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_test_images.bin");
        exit(1);
    }
    
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberImages, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberRows, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberCols, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);


    // printf("%d \n" , (int)numberRows);
    for (int i = 0; i<NUMBER_TRAINING_DATA; i++)
    {
        for (int j = 0; j<INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread_s(&pixel, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTraining);
            trainingData[i][j] = (double) pixel;
        }
    }

    for (int i = 0; i<NUMBER_TEST_DATA; i++)
    {
        for (int j = 0; j<INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread_s(&pixel, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTest);
            testData[i][j] = (double) pixel;
        }
    }
    printf("done reading images");
    // print some training samples
    printSampleImages();

    fclose(fileStreamTraining);
    fclose(fileStreamTest);
}


void readMNISTLabelData()
{
    FILE* fileStreamTraining;
    FILE* fileStreamTest;
    errno_t error;
    if ((error = fopen_s(&fileStreamTraining, "./mnist_train_labels.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_train_labels.bin");
        exit(1);
    }
    
    int32_t magicNumber;
    int32_t numberLabels; 
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);
    fread_s(&numberLabels, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTraining);

    if ((error = fopen_s(&fileStreamTest, "./mnist_test_labels.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_test_labels.bin");
        exit(1);
    }
    
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberLabels, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);

    for (int i = 0; i<NUMBER_TRAINING_DATA; i++)
    {
        unsigned char label;
        fread_s(&label, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTraining);
        trainingLabels[i] = (int)label;

    }

    for (int i = 0; i<NUMBER_TEST_DATA; i++)
    {
        unsigned char label;
        fread_s(&label, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTest);
        testLabels[i] = (int)label;
    }

    // print some training samples
    printSampleLabels();
    printf("done reading labels");
    fclose(fileStreamTraining);
    fclose(fileStreamTest);
}

int main() {

    readMNISTImageData();
    readMNISTLabelData();


    return 0;
}