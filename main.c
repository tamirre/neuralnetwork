#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))

typedef struct NeuralNetwork {
    int numberInputNodes;
    int numberOutputNodes;
    int numberLayers;
    int numberMaxEpochs;
    int batchSize;
    double gradientTolerance;
    double learningRate;
    double*** weights;
    double** biases;
    double** z;
    int* layerSizes;
    double** activations;
    double** deltas;
} NeuralNetwork;

typedef struct Data {
    int numberTrainingLabels;
    int numberTestLabels;
    int numberTrainingData;
    int numberTestData;

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

void printSampleImages(Data* data, NeuralNetwork* neuralNetwork)
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < neuralNetwork->numberInputNodes; j++)
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
        for (int j = 0; j < neuralNetwork->numberInputNodes; j++)
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

void freeMemory(Data* data, NeuralNetwork* neuralNetwork)
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
    
    for (int layer = 0; layer < neuralNetwork->numberLayers - 1; layer++) 
    {
        for (int i = 0; i < neuralNetwork->layerSizes[layer]; i++)
        {
            free(neuralNetwork->weights[layer][i]);
        }
        free(neuralNetwork->weights[layer]);
        free(neuralNetwork->biases[layer]);
        free(neuralNetwork->activations[layer]);
        free(neuralNetwork->deltas[layer]);
        free(neuralNetwork->z[layer]);
    }

    free(neuralNetwork->weights);
    free(neuralNetwork->biases);
    free(neuralNetwork->activations);
    free(neuralNetwork->deltas);
    free(neuralNetwork->z);
    free(neuralNetwork->layerSizes);
}

int readMNISTImageData(Data* data, NeuralNetwork* neuralNetwork)
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

    // Todo: choose datatype according to magic number:
    // The magic number is an integer (MSB first). The first 2 bytes are always 0.
    // The third byte codes the type of the data:
    // 0x08: unsigned byte
    // 0x09: signed byte
    // 0x0B: short (2 bytes)
    // 0x0C: int (4 bytes)
    // 0x0D: float (4 bytes)
    // 0x0E: double (8 bytes)
    // The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
    // The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

    // const int32_t magicNumberBigEndianTraining = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberImagesBigEndianTraining = intBigEndianToLittleEndian(numberImages);
    const int32_t numberRowsBigEndianTraining = intBigEndianToLittleEndian(numberRows);
    const int32_t numberColsBigEndianTraining = intBigEndianToLittleEndian(numberCols);

    neuralNetwork->numberInputNodes = numberRowsBigEndianTraining * numberColsBigEndianTraining;
    data->numberTrainingData = numberImagesBigEndianTraining;
    
    data->trainingData = malloc(data->numberTrainingData * sizeof(double*));
    if (data->trainingData == NULL)
    {
        perror("Failed to allocate memory for trianingData!");
        return 1;
    }
    for (int i = 0; i < data->numberTrainingData; i++) {
        data->trainingData[i] = malloc(neuralNetwork->numberInputNodes * sizeof(double));
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
        for (int j = 0; j < neuralNetwork->numberInputNodes; j++)
        {
            unsigned char pixel;
            fread_s(&pixel, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTraining);
            data->trainingData[i][j] = (double)pixel / 255.0; // normalize to values to be in [0,1]
        }
    }

    if ((error = fopen_s(&fileStreamTest, "./mnist_test_images.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_test_images.bin");
        exit(1);
    }
    
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberImages, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberRows, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberCols, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);

    // const int32_t magicNumberBigEndianTest = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberImagesBigEndianTest = intBigEndianToLittleEndian(numberImages);

    data->numberTestData = numberImagesBigEndianTest;
    data->testData = malloc(data->numberTestData * sizeof(double*));
    if (data->testData == NULL)
    {
        perror("Failed to allocate memory for testData!");
        return 1;
    }
    for (int i = 0; i < data->numberTestData; i++) {
        data->testData[i] = malloc(neuralNetwork->numberInputNodes * sizeof(double));
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
        for (int j = 0; j < neuralNetwork->numberInputNodes; j++)
        {
            unsigned char pixel;
            fread_s(&pixel, sizeof(unsigned char), sizeof(unsigned char), 1, fileStreamTest);
            data->testData[i][j] = (double) pixel / 255.0; // normalize to values in [0,1];
        }
    }
    
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

    // const int32_t magicNumberBigEndian = intBigEndianToLittleEndian(magicNumber);
    const int32_t numberLabelsBigEndian = intBigEndianToLittleEndian(numberLabels);
    
    data->numberTrainingLabels = numberLabelsBigEndian;
    data->trainingLabels = malloc(data->numberTrainingLabels * sizeof(int));
    if (data->trainingLabels == NULL)
    {
        perror("Failed to allocate memory for testData!");
        return 1;
    }

    if ((error = fopen_s(&fileStreamTest, "./mnist_test_labels.bin", "rb")) != 0 )
    {
        printf("ERROR: could not read file: ./mnist_test_labels.bin");
        exit(1);
    }
    
    fread_s(&magicNumber, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);
    fread_s(&numberLabels, sizeof(int32_t), sizeof(int32_t), 1, fileStreamTest);

    // const int32_t magicNumberBigEndianTraining = intBigEndianToLittleEndian(magicNumber);
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
    
    fclose(fileStreamTraining);
    fclose(fileStreamTest);
    return 0;
}

double random_value(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

void printWeightsAndBiases(NeuralNetwork* neuralNetwork)
{
    for (int layer = 0; layer < neuralNetwork->numberLayers - 1; layer++) {
        printf("Layer %d to %d weights:\n", layer, layer + 1);
        for (int i = 0; i < neuralNetwork->layerSizes[layer]; i++) {
            for (int j = 0; j < neuralNetwork->layerSizes[layer + 1]; j++) {
                printf("%.2f ", neuralNetwork->weights[layer][i][j]);
            }
            printf("\n");
        }
        printf("Layer %d biases:\n", layer + 1);
        for (int j = 0; j < neuralNetwork->layerSizes[layer + 1]; j++) {
            printf("%.2f ", neuralNetwork->biases[layer][j]);
        }
        printf("\n");
    }

    for (int layer = 0; layer < neuralNetwork->numberLayers; layer++) {
        double sum = 0;
        printf("\n");
        printf("Layer %d pre-activation z:\n", layer);
        for (int j = 0; j < neuralNetwork->layerSizes[layer]; j++) {
            printf("%.4f ", neuralNetwork->z[layer][j]);
            sum += neuralNetwork->z[layer][j];
        }
        printf("\n");
        printf("sum = %f\n", sum);
    }

    for (int layer = 0; layer < neuralNetwork->numberLayers; layer++) {
        double sum = 0;
        printf("\n");
        printf("Layer %d activations:\n", layer);
        for (int j = 0; j < neuralNetwork->layerSizes[layer]; j++) {
            printf("%.4f ", neuralNetwork->activations[layer][j]);
            sum += neuralNetwork->activations[layer][j];
        }
        printf("\n");
        printf("sum = %f\n", sum);
    }
}

void softmax(double *z, double *activations, int size) 
{
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        activations[i] = exp(z[i]); 
        sum += activations[i];
    }
    for (int i = 0; i < size; i++) {
        activations[i] /= sum; // Normalize to make it a probability distribution
    }
}

double sigmoid(double x)
{
    return (1.0 / (1.0 + exp(-x)));
}

double sigmoidPrime(double x)
{
    return  sigmoid(x) * (1 - sigmoid(x));
}

void forwardPass(double **data, NeuralNetwork* neuralNetwork, int inputIndex)
{
    // initialize input layer activations to the input
    for (int i = 0; i < neuralNetwork->layerSizes[0]; i++) 
    {
        neuralNetwork->activations[0][i] = data[inputIndex][i];
    }
    for (int layer = 1; layer < neuralNetwork->numberLayers; layer++) 
    {
        int inputSize = neuralNetwork->layerSizes[layer-1];
        int outputSize = neuralNetwork->layerSizes[layer];
        
        // Calculate z = W * a + b
        for (int j = 0; j < outputSize; j++) 
        {
            neuralNetwork->z[layer][j] = neuralNetwork->biases[layer][j];
            for (int i = 0; i < inputSize; i++) 
            {
                neuralNetwork->z[layer][j] += neuralNetwork->weights[layer-1][i][j] 
                                           * neuralNetwork->activations[layer-1][i];
            }
        }
        if (layer == neuralNetwork->numberLayers - 1)
        {
            softmax(neuralNetwork->z[layer], neuralNetwork->activations[layer], outputSize);
        } 
        else
        {
            for (int j = 0; j < outputSize; j++) 
            {
                neuralNetwork->activations[layer][j] = sigmoid(neuralNetwork->z[layer][j]);
            }
        }
    }
}

void labelOutput(int label, int numClasses, double* encodedOutput)
{
    for (int i = 0; i < numClasses; i++) {
        encodedOutput[i] = (i == label) ? 1.0 : 0.0;
    }
}

double costFunctionDerivative(double activation, double target)
{
    return (activation - target);
}

double backPropagation(Data* data, NeuralNetwork* neuralNetwork, int imgIndex)
{
    int lastLayer = neuralNetwork->numberLayers - 1;
    double* encodedOutput = (double *)malloc(neuralNetwork->layerSizes[lastLayer] * sizeof(double));
    labelOutput(data->trainingLabels[imgIndex], neuralNetwork->layerSizes[lastLayer], encodedOutput);

    // Compute delta for output layer
    for(int i = 0; i < neuralNetwork->layerSizes[lastLayer]; i++)
    {
        neuralNetwork->deltas[lastLayer][i] = costFunctionDerivative(neuralNetwork->activations[lastLayer][i], encodedOutput[i]) 
                                            * sigmoidPrime(neuralNetwork->z[lastLayer][i]);
    }
    free(encodedOutput);

    // Compute delta backwards for every hidden layer through the network
    for(int layer = neuralNetwork->numberLayers - 2; layer > 0; layer--)
    {
        int inputSize = neuralNetwork->layerSizes[layer];
        int outputSize = neuralNetwork->layerSizes[layer + 1];

        for (int i = 0; i < inputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < outputSize; j++) {
                sum += neuralNetwork->deltas[layer + 1][j] * neuralNetwork->weights[layer][i][j];
            }
            neuralNetwork->deltas[layer][i] = sum * sigmoidPrime(neuralNetwork->z[layer][i]);
        }
    }

    // Update weights and biases
    double gradientNorm = 0.0;
    for(int layer = neuralNetwork->numberLayers - 1; layer > 0; layer--)
    {
        int inputSize = neuralNetwork->layerSizes[layer-1];
        int outputSize = neuralNetwork->layerSizes[layer];

        for (int j = 0; j < outputSize; j++) {
            neuralNetwork->biases[layer][j] -= neuralNetwork->learningRate * neuralNetwork->deltas[layer][j];
            for (int i = 0; i < inputSize; i++) {
                double grad = neuralNetwork->deltas[layer][j] * neuralNetwork->activations[layer-1][i];
                neuralNetwork->weights[layer-1][i][j] -= neuralNetwork->learningRate * grad;
                gradientNorm += grad * grad;
            }
        }
    }
    return gradientNorm;
}

int initNeuralNetwork(NeuralNetwork* neuralNetwork,  int *layerSizes)
{
    neuralNetwork->layerSizes = (int *)malloc(neuralNetwork->numberLayers * sizeof(int));

    for (int i = 0; i < neuralNetwork->numberLayers; i++) {
        neuralNetwork->layerSizes[i] = layerSizes[i];
    }

    // Allocate memory for weights and biases
    neuralNetwork->weights = (double ***)malloc((neuralNetwork->numberLayers - 2) * sizeof(double **));
    neuralNetwork->biases = (double **)malloc((neuralNetwork->numberLayers - 1) * sizeof(double *));
    neuralNetwork->z = (double **)malloc((neuralNetwork->numberLayers - 1) * sizeof(double *));
    neuralNetwork->activations = (double **)malloc((neuralNetwork->numberLayers - 1) * sizeof(double *));
    neuralNetwork->deltas = (double **)malloc((neuralNetwork->numberLayers - 1) * sizeof(double *));

    for (int layer = 0; layer < neuralNetwork->numberLayers-1; layer++) 
    {
        int inputSize = neuralNetwork->layerSizes[layer];
        int outputSize = neuralNetwork->layerSizes[layer + 1];

        // Allocate weights for the current layer
        neuralNetwork->weights[layer] = (double **)malloc(inputSize * sizeof(double *));
        for (int i = 0; i < inputSize; i++) {
            neuralNetwork->weights[layer][i] = (double *)malloc(outputSize * sizeof(double));
            for (int j = 0; j < outputSize; j++) {
                neuralNetwork->weights[layer][i][j] = random_value(- sqrt(6.0 / inputSize), sqrt(6.0 / inputSize)); // Xavier initialization
            }
        }
    }

    for (int layer = 0; layer < neuralNetwork->numberLayers; layer++) {
        neuralNetwork->biases[layer] = (double *)malloc(neuralNetwork->layerSizes[layer] * sizeof(double));
        neuralNetwork->activations[layer] = (double *)malloc(neuralNetwork->layerSizes[layer] * sizeof(double));
        neuralNetwork->deltas[layer] = (double *)malloc(neuralNetwork->layerSizes[layer] * sizeof(double));
        neuralNetwork->z[layer] = (double *)malloc(neuralNetwork->layerSizes[layer] * sizeof(double));
        for (int j = 0; j < neuralNetwork->layerSizes[layer]; j++) {
            neuralNetwork->biases[layer][j] = random_value(- sqrt(6.0 / neuralNetwork->layerSizes[layer]), sqrt(6.0 / neuralNetwork->layerSizes[layer])); // Xavier initialization 
            neuralNetwork->activations[layer][j] = 0.0; // Initialize activations
            neuralNetwork->deltas[layer][j] = 0.0; // Initialize deltas
            neuralNetwork->z[layer][j] = 0.0;
        }
    }
    return 0;
}

void testNeuralNetwork(Data* data, NeuralNetwork* neuralNetwork) {
    int correctPredictions = 0;
    for (int i = 0; i < data->numberTestData; i++) {
        // Perform a forward pass on the test data
        forwardPass(data->testData, neuralNetwork, i);
        // Get the predicted label (index of the highest activation)
        int predictedLabel = -1;
        double maxActivation = -1.0;
        for (int j = 0; j < neuralNetwork->layerSizes[neuralNetwork->numberLayers - 1]; j++) {
            if (neuralNetwork->activations[neuralNetwork->numberLayers - 1][j] > maxActivation) {
                maxActivation = neuralNetwork->activations[neuralNetwork->numberLayers - 1][j];
                predictedLabel = j;
            }
        }

        // Check if the prediction is correct
        if (predictedLabel == data->testLabels[i]) {
            correctPredictions++;
        }
    }

    // Calculate and print accuracy
    double accuracy = (double)correctPredictions / data->numberTestData * 100.0;
    printf("Test accuracy: %.2f%%\n", accuracy);
}

void trainNeuralNetwork(Data* data, NeuralNetwork* neuralNetwork)
{
    int epoch = 0;
    while (epoch < neuralNetwork->numberMaxEpochs)
    {
        double gradientNorm = 0;
        printf("========= EPOCH %d =========\n", epoch);    
        for (int imgIndex = 0; imgIndex < data->numberTrainingData; imgIndex++)
        {
            forwardPass(data->trainingData, neuralNetwork, imgIndex);
            gradientNorm += backPropagation(data, neuralNetwork, imgIndex); 
        }
        gradientNorm = sqrt(gradientNorm) / data->numberTrainingData;
        printf("gradient: %.4e\n", gradientNorm);
        if(gradientNorm < neuralNetwork->gradientTolerance) break;
        epoch++;
    }
    printf("==== TRAINING FINISHED ====\n");
}




int main() {

    // Initialize random seed
    // unsigned int seedTime = (unsigned int)time(NULL);
    // srand(seedTime);
    srand(42); // Deterministic seeding

    // Initialize data and neural network
    Data data;
    NeuralNetwork neuralNetwork = {
        .numberOutputNodes = 10,
        .numberLayers = 3,
        .numberMaxEpochs = 3,
        .learningRate = 0.1,
        .gradientTolerance = 1e-6,
    };

    // Read training & test image data anPd labels, exit on failure
    if(readMNISTImageData(&data, &neuralNetwork)) return 1;
    if(readMNISTLabelData(&data)) return 1;
    printf("Reading image and label data complete!\n");

    neuralNetwork.batchSize = 1; // data.numberTrainingData / 100;

    int layerSizes[3] = {neuralNetwork.numberInputNodes, 100, neuralNetwork.numberOutputNodes};
    initNeuralNetwork(&neuralNetwork, layerSizes);
    trainNeuralNetwork(&data, &neuralNetwork);
    testNeuralNetwork(&data, &neuralNetwork);

    // Free allocated memory
    freeMemory(&data, &neuralNetwork);
    return 0;
}