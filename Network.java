    public class Main {
        public static void main(String[] args) {
            network test = new network(2, new int[] {8, 8, 1 });

            double[][] inputs = {
                    {0.0, 0.0},
                    {0.0, 1.0},
                    {1.0, 0.0},
                    {1.0, 1.0}
            };

            double[][] expectedOutputs = {
                    {0.0, 1.0},
                    {1.0, 0.0},
                    {1.0, 0.0},
                    {0.0, 1.0}
            };


            for (int i = 0; i < 10000; i++) {
                for (int n = 0; n < inputs.length; n++){
                    double result = test.Train(inputs[n], expectedOutputs[n]);
                    System.out.println(result);
                }
            }
            System.out.println("-------");
            System.out.println(test.run(new double[] {1, 1})[0]);
        }
    }

    class neuron {
        public double[] Weights;
        public double Bias;

        public double NetValue = 0;
        public double ActivationValue = 0;

        public double Adjustment = 0;

        public double[] Momentum;
        public double MomentumB;
        public double[] Velocity;
        public double VelocityB;

        public double time = 1;

        public neuron(int Connections) {
            Weights = new double[Connections];
            Momentum = new double[Connections];
            Velocity = new double[Connections];
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
            ActivationFunction();
        }

        private void ActivationFunction() {
            double x = Math.exp(NetValue);
            ActivationValue = x / (x + 1);
        }

        public double ActivationFunctionDerivative() {
            double x = Math.exp(NetValue);
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

        public static final double LearnRate = 0.02;
        public static final double FirstDecay = 0.9;
        public static final double SecondDecay = 0.999;
        public static final double WeightDecay = 0.01;
        public static final double Correction = 1E-8;

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
                Ne.Adjustment = Ne.ActivationFunctionDerivative() * costDerivatives[j];
            }

            // Calculate the rest
            for (int l = layerLength - 2; l >= 0; l--) {
                for (int k = 0; k < Layers[l].Neurons.length; k++) {
                    neuron Ne = Layers[l].Neurons[k];
                    Ne.Adjustment = 0;
                    for (int w = 0; w < Layers[l + 1].Neurons.length; w++) {
                        Ne.Adjustment += Layers[l + 1].Neurons[w].Weights[k] * Layers[l + 1].Neurons[w].Adjustment;
                    }
                    Ne.Adjustment *= Ne.ActivationFunctionDerivative();
                }
            }
        }

        public void CalculateGradient(double[] input) {
            double[] ActivationValues = input;

            for (int l = 0; l < Layers.length; l++) {

                layer currentLayer = Layers[l];

                if (l > 0) {
                    neuron[] LastNeurons = Layers[l - 1].Neurons;
                    ActivationValues = new double[LastNeurons.length];
                    for (int n = 0; n < LastNeurons.length; n++) {
                        ActivationValues[n] = LastNeurons[n].ActivationValue;
                    }
                }

                for (int n = 0; n < currentLayer.Neurons.length; n++) {

                    neuron Ne = currentLayer.Neurons[n];

                    for (int w = 0; w < Ne.Weights.length; w++) {
                        double gradient = ActivationValues[w] * Ne.Adjustment;
                        //Moment calculations
                        Ne.Momentum[w] = FirstDecay * Ne.Momentum[w] + (1 - FirstDecay) * gradient;
                        Ne.Velocity[w] = SecondDecay * Ne.Velocity[w] + (1 - SecondDecay) * Math.pow(gradient, 2);
                        //Bias Correction
                        double CrrMomentum = Ne.Momentum[w] / (1 - Math.pow(1 - FirstDecay, Ne.time));
                        double CrrVelocity = Ne.Velocity[w] / (1 - Math.pow(1 - SecondDecay, Ne.time));
                        //Update weights
                        Ne.Weights[w] = Ne.Weights[w] - LearnRate * CrrMomentum / (Math.sqrt(CrrVelocity) + Correction);
                        //weight decay???
                        Ne.Weights[w] -= LearnRate * WeightDecay * Ne.Weights[w];

                    }
                    double gradient = Ne.Adjustment;
                    //Moment calculations
                    Ne.MomentumB = FirstDecay * Ne.MomentumB + (1 - FirstDecay) * gradient;
                    Ne.VelocityB = SecondDecay * Ne.VelocityB + (1 - SecondDecay) * Math.pow(gradient, 2);
                    //Bias Correction
                    double CrrMomentum = Ne.MomentumB / (1 - Math.pow(1 - FirstDecay, Ne.time));
                    double CrrVelocity = Ne.VelocityB / (1 - Math.pow(1 - SecondDecay, Ne.time));
                    //Update weights
                    Ne.Bias = Ne.Bias - LearnRate * CrrMomentum / (Math.sqrt(CrrVelocity) + Correction);
                    //weight decay???
                    Ne.Bias -= LearnRate * WeightDecay * Ne.Bias;

                    Ne.time++;
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
