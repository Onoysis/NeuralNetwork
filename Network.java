public class Main {
	public static void main(String[] args) {
		network test = new network(2, new int[] { 3, 1 });

		double[] input = new double[2];
		double[] output = new double[1];

		for (int i = 0; i < 5000; i++) {
			input[0] = Math.random();
			input[1] = Math.random();

			output[0] = input[1] > input[0] ? 1 : 0;

			double result = test.Train(input, output);
			System.out.println(result);
		}
		System.out.println("-------");
		System.out.println(test.run(new double[] { 0.5, 0.7 })[0]);
	}
}

class neuron {
	public double[] Weights;
	public double Bias;

	public double NetValue = 0;
	public double ActivationValue = 0;

	public double Adjustment = 0;

	public neuron(int Connections) {
		Weights = new double[Connections];
		for (int i = 0; i < Connections; i++) {
			Weights[i] = Math.random() * 2 - 1;
		}
		Bias = Math.random() * 2 - 1;
	}

	public void fire(double[] Values) {
		NetValue = 0;
		ActivationValue = 0;

		int Count = 0;

		for (double num : Values) {
			NetValue += num * Weights[Count++];
		}
		NetValue += Bias;
		ActivationValue = neuron.ActivationFunction(NetValue);
	}

	private static double ActivationFunction(double value) {
		double x = Math.exp(value);
		return x / (x + 1);
	}

	public static double ActivationFunctionDerivative(double value) {
		double x = Math.exp(value);
		return x / Math.pow(x + 1, 2);
	}
}

class layer {
	public neuron[] Neurons;

	public layer(int n, int c) {
		Neurons = new neuron[n];
		for (int i = 0; i < n; i++) {
			Neurons[i] = new neuron(c);
		}
	}

	public void activate(double[] input) {
		for (neuron node : Neurons) {
			node.fire(input);
		}
	}
}

class network {

	public static final double LearnRate = 0.2;

	public layer[] Layers;

	public network(int inputLayer, int[] layers) {

		Layers = new layer[layers.length];

		int LayerSize = inputLayer;

		int count = 0;

		for (int i : layers) {
			Layers[count++] = new layer(i, LayerSize);
			LayerSize = i;
		}
	}

	public double[] run(double[] input) {
		double[] prev_result = input;
		for (layer Layer : Layers) {
			Layer.activate(prev_result);

			double[] new_prev_result = new double[Layer.Neurons.length];

			for (int d = 0; d < new_prev_result.length; d++) {
				new_prev_result[d] = Layer.Neurons[d].ActivationValue;
			}
			prev_result = new_prev_result;
		}
		return prev_result;
	}

	public void CalculateAdjustment(double[] costDerivatives) {
		int layerLength = Layers.length;

		// Calculate first layer
		for (int j = 0; j < Layers[layerLength - 1].Neurons.length; j++) {
			neuron Ne = Layers[layerLength - 1].Neurons[j];
			Ne.Adjustment = neuron.ActivationFunctionDerivative(Ne.NetValue) * costDerivatives[j];
		}

		// Calculate the rest
		for (int l = layerLength - 2; l >= 0; l--) {
			for (int k = 0; k < Layers[l].Neurons.length; k++) {
				neuron Ne = Layers[l].Neurons[k];
				Ne.Adjustment = 0;
				for (int w = 0; w < Layers[l + 1].Neurons.length; w++) {
					Ne.Adjustment += Layers[l + 1].Neurons[w].Weights[k] * Layers[l + 1].Neurons[w].Adjustment;
				}
				Ne.Adjustment *= neuron.ActivationFunctionDerivative(Ne.NetValue);
			}
		}
	}

	public void CalculateGradient(double[] input) {

		for (int i = 0; i < Layers[0].Neurons.length; i++) {
			neuron Ne = Layers[0].Neurons[i];

			for (int w = 0; w < Ne.Weights.length; w++) {

				Ne.Weights[w] -= input[w] * LearnRate * Ne.Adjustment;
			}

			Ne.Bias -= Ne.Adjustment * LearnRate;
		}

		for (int l = 1; l < Layers.length; l++) {

			layer currentLayer = Layers[l];

			neuron[] LastNeurons = Layers[l - 1].Neurons;

			double[] ActivationValues = new double[LastNeurons.length];

			for (int n = 0; n < LastNeurons.length; n++) {
				ActivationValues[n] = LastNeurons[n].NetValue;
			}

			for (int n = 0; n < currentLayer.Neurons.length; n++) {

				neuron Ne = currentLayer.Neurons[n];

				for (int w = 0; w < Ne.Weights.length; w++) {
					Ne.Weights[w] -= ActivationValues[w] * LearnRate * Ne.Adjustment;
				}

				Ne.Bias -= Ne.Adjustment * LearnRate;
			}
		}
	}

	public double Train(double[] input, double[] SearchedOutput) {
		double[] output = run(input);

		double[] costDerivatives = new double[SearchedOutput.length];

		for (int o = 0; o < output.length; o++) {
			costDerivatives[o] = 2 * (output[o] - SearchedOutput[o]);
		}

		CalculateAdjustment(costDerivatives);
		CalculateGradient(input);

		double cost = 0;

		for (int o = 0; o < output.length; o++) {
			cost += Math.pow(SearchedOutput[o] - output[o], 2);
		}

		return cost;
	}
}
