import java.util.Random;

public class NeuralNetwork {
    private int inputNodes;
    private int hiddenNodes;
    private int outputNodes;
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double learningRate;
    
    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;

        weightsInputHidden = new double[inputNodes][hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes][outputNodes];
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() - 0.5; // Инициализация весов случайными значениями
            }
        }
        for (int j = 0; j < hiddenNodes; j++) {
            for (int k = 0; k < outputNodes; k++) {
                weightsHiddenOutput[j][k] = rand.nextDouble() - 0.5; // Инициализация весов случайными значениями
            }
        }
    }

    public double[] predict(double[] input) {
        double[] hidden = new double[hiddenNodes];
        double[] output = new double[outputNodes];
        
        // Прямое распространение (Forward pass)
        for (int j = 0; j < hiddenNodes; j++) {
            hidden[j] = 0;
            for (int i = 0; i < inputNodes; i++) {
                hidden[j] += input[i] * weightsInputHidden[i][j];
            }
            hidden[j] = sigmoid(hidden[j]);
        }

        for (int k = 0; k < outputNodes; k++) {
            output[k] = 0;
            for (int j = 0; j < hiddenNodes; j++) {
                output[k] += hidden[j] * weightsHiddenOutput[j][k];
            }
            output[k] = sigmoid(output[k]);
        }

        return output;
    }

    public void train(double[] input, double[] target) {
        // Прямое распространение для получения скрытых и выходных значений
        double[] hidden = new double[hiddenNodes];
        double[] output = predict(input);

        // Ошибка выхода
        double[] outputErrors = new double[outputNodes];
        for (int i = 0; i < outputNodes; i++) {
            outputErrors[i] = target[i] - output[i];
        }

        // Ошибка скрытого слоя
        double[] hiddenErrors = new double[hiddenNodes];
        for (int j = 0; j < hiddenNodes; j++) {
            hiddenErrors[j] = 0;
            for (int k = 0; k < outputNodes; k++) {
                hiddenErrors[j] += outputErrors[k] * weightsHiddenOutput[j][k];
            }
        }

        // Обновление весов для выходного слоя
        for (int j = 0; j < hiddenNodes; j++) {
            for (int k = 0; k < outputNodes; k++) {
                weightsHiddenOutput[j][k] += learningRate * outputErrors[k] * hidden[j];
            }
        }

        // Обновление весов для скрытого слоя
        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * input[i];
            }
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
}
