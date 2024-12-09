public class SingleLayerPerceptron {
    private int inputNodes;
    private int outputNodes;
    private double[][] weights;
    private double learningRate;
    
    public SingleLayerPerceptron(int inputNodes, int outputNodes, double learningRate) {
        this.inputNodes = inputNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        weights = new double[inputNodes][outputNodes];
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < outputNodes; j++) {
                weights[i][j] = rand.nextDouble() - 0.5;
            }
        }
    }

    public double[] predict(double[] input) {
        double[] output = new double[outputNodes];
        for (int j = 0; j < outputNodes; j++) {
            output[j] = 0;
            for (int i = 0; i < inputNodes; i++) {
                output[j] += input[i] * weights[i][j];
            }
            output[j] = sigmoid(output[j]);
        }
        return output;
    }

    public void train(double[] input, double[] target) {
        double[] output = predict(input);
        for (int j = 0; j < outputNodes; j++) {
            double error = target[j] - output[j];
            for (int i = 0; i < inputNodes; i++) {
                weights[i][j] += learningRate * error * input[i];
            }
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
