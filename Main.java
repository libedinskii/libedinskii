import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        // Определите количество входов, скрытых и выходных узлов
        int inputNodes = 784; // 28x28 изображение
        int hiddenNodes = 128; // Количество скрытых узлов
        int outputNodes = 10; // Цифры от 0 до 9
        double learningRate = 0.1;

        NeuralNetwork nn = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        SingleLayerPerceptron slp = new SingleLayerPerceptron(inputNodes, outputNodes, learningRate);

        // Пример данных (или загрузите MNIST)
        double[][] trainingData = {}; // Ваши данные для обучения
        double[][] trainingLabels = {}; // Метки для обучения
        
        // Обучайте нейронную сеть и перцептрон
        for (int epoch = 0; epoch < 5; epoch++) {
            for (int i = 0; i < trainingData.length; i++) {
                nn.train(trainingData[i], trainingLabels[i]);
                slp.train(trainingData[i], trainingLabels[i]);
            }
        }

        System.out.println("Обучение завершено.");
    }
}
