using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class Network
    {
        Layer input;
        DerivedLayer[] hiddenLayers;
        DerivedLayer output;
        IndexOutOfRangeException DeReLuFuckedUpAgain = new IndexOutOfRangeException();
        DataMisalignedException DeReLuFuckedEverythingUpAgain = new DataMisalignedException();

        double learningFactor = -0.00001; //the lower this is, the slower but more accurately the network learns

        /// <summary>
        /// Creates a network using the passed vector as input.
        /// </summary>
        /// <param name="data"></param>
        public Network(double[] data)
        {
            input = new Layer(data);
        }

        /// <summary>
        /// Creates a network that takes inputSize neurons
        /// </summary>
        /// <param name="inputSize"></param>
        /// <param name="hiddenSizes"></param>
        /// <param name="outputSize"></param>
        public Network(int inputSize, int[] hiddenSizes, int outputSize)
        {
            input = new Layer(new double[inputSize]); //input layer is your data
            if (hiddenSizes.Count() > 0)
            {
                hiddenLayers = new DerivedLayer[hiddenSizes.Count()]; //where the stuff happens
                hiddenLayers[0] = new DerivedLayer(input, hiddenSizes[0]); //first hidden layer pulls from input
                for (int i = 1; i < hiddenSizes.Count(); i++)
                {
                    hiddenLayers[i] = new DerivedLayer(hiddenLayers[i - 1], hiddenSizes[i]); //then the next hidden layers pull from the hidden layer before them
                }
                output = new DerivedLayer(hiddenLayers[hiddenLayers.GetUpperBound(0)], outputSize); //the output is derived from the last hidden layer
            }
            else
            {
                output = new DerivedLayer(input, outputSize); //directly from input to output (probably sucks)
            }
        }

        /// <summary>
        /// Runs the network on a set of input data (image).
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public Vector<double> eval(Vector<double> data)
        {
            input.neurons = data; //give data to input layer
            for(int i = 0; i < hiddenLayers.Count(); i++)
            {
                hiddenLayers[i].deriveValues(); //each hidden layer calculates its values from the last hidden layer, or input for the first hidden layer
            }
            output.deriveValues(); //get output from final hidden layer
            return output.neurons; //return activations of each output neuron
        }

        public Vector<double> eval(double[] adata)
        {
            Vector<double> data = Vector<double>.Build.Dense(adata); //same thing as above, but parses an array of doubles into a vector
            input.neurons = data; //give data to input layer
            for (int i = 0; i < hiddenLayers.Count(); i++)
            {
                hiddenLayers[i].deriveValues(); //each hidden layer calculates its values from the last hidden layer, or input for the first hidden layer
            }
            output.deriveValues(); //get output from final hidden layer
            return output.neurons; //return activations of each output neuron
        }

        public void randomizeAll() //initialize network with random values
        {
            for (int i = 0; i < hiddenLayers.Count(); i++)
            {
                hiddenLayers[i].randomize();
            }
            output.randomize();
        }

        public double numFactors()
        {
            double ret = 0;
            for(int i = 0; i < hiddenLayers.Count(); i++)
            {
                ret += hiddenLayers[i].numFactors();
            }
            return ret + output.numFactors();
        }

        public double costFun(double[] expected)
        {
            Vector<double> input = Vector<double>.Build.Dense(expected); //to hold expected valuese
            return Math.Pow((input - output.neurons).L2Norm(), 2)/input.Count; //l2norm gives square root of sum of sqares of errors, square just gets the sum, then dividing gives avg of squared errors
        }

        public double check(double[] expected)
        {
            double temp = costFun(expected);
            int expectedIndex = Array.IndexOf(expected, 1);
            if (temp == 0.14285714285714285) // this is 1/7 in double limit
            {
                throw DeReLuFuckedEverythingUpAgain; //this means it's fucked up and put every output to 0, so this should make it mix up the weights and biases
            }
            else if (temp < 1 && output.neurons[expectedIndex] == 0)
            {
                throw DeReLuFuckedUpAgain; //if the error is low because expected output is 0 and other activations are low
            }
            else
            {
                return temp; //working as expected
            }
        }

        public GradStep[] backpropogate(double[] expected)
        {
            double[][] errorPerActivation = new double[hiddenLayers.Count() + 1][]; //change in error per change in activation, multiplied by change in activation per change in weight/bias to give gradient
            GradStep[] steps = new GradStep[hiddenLayers.Count() + 1]; //the changes in weights and biases for each hidden layer, plus the output

            //Calculate changes for final layer
            errorPerActivation[hiddenLayers.Count()] = new double[output.weights.RowCount]; //have to initialize each array because it's jagged and c# is the dicks
            steps[steps.Count() - 1] = new GradStep(output.weights.RowCount, output.weights.ColumnCount, output.neurons.Count); //initialize gradstep for the output
            for (int i = 0; i < output.weights.RowCount; i++)
            {
                errorPerActivation[hiddenLayers.Count()][i] = 2 * (expected[i] - output.neurons[i]) * Compressions.derReLU(output.neurons[i]); //change in error per change in activation for the final layer
                for (int j = 0; j < output.weights.ColumnCount; j++)
                {
                    steps[steps.Count() - 1].weightStep[i, j] = errorPerActivation[hiddenLayers.Count()][i] * hiddenLayers[hiddenLayers.Count() - 1].neurons[j]; //the change in weight is the error/activation times activation/weight
                }
                steps[steps.Count() - 1].biasStep[i] = errorPerActivation[hiddenLayers.Count()][i]; //change in error with respect to bias is just error/activation, since activation/bias is 1 (bias is constant)
            }

            for(int i = hiddenLayers.Count() - 1; i >= 0; i--) //for every hidden layer, iterating backwards since we need the next layer's error
            {
                DerivedLayer nextLayer;
                Layer previousLayer;
                errorPerActivation[i] = new double[hiddenLayers[i].weights.RowCount]; //have to initialize each array because it's jagged and c# is the balls
                steps[i] = new GradStep(hiddenLayers[i].weights.RowCount, hiddenLayers[i].weights.ColumnCount, hiddenLayers[i].biases.Count); //holds suggested changes to weights and biases in this layer

                if(i < hiddenLayers.Count() - 1)
                {
                    nextLayer = hiddenLayers[i + 1]; //use values from next hidden layer
                }
                else
                {
                    nextLayer = output; //unless it's the final hidden layer, in which case we need values of output
                }

                if (i > 0)
                {
                    previousLayer = hiddenLayers[i - 1]; //use values from previous hidden layer
                }
                else
                {
                    previousLayer = input; //unless it's the final hidden layer, in which case we need values of input
                }

                for (int j = 0; j < hiddenLayers[i].weights.RowCount; j++) //in every row
                {
                    double activation = Compressions.derReLU(hiddenLayers[i].neurons[j]); //find the change in error per change in activation
                    errorPerActivation[i][j] = 0;
                    for (int k = 0; k < nextLayer.neurons.Count(); k++) //since this activation affects every neuron in next layer, we need changes for all of them
                    {
                        errorPerActivation[i][j] += nextLayer.weights[k, j] * errorPerActivation[i + 1][k]; //change in activation per weight times change in error per activation
                        steps[i].biasStep[j] += errorPerActivation[i + 1][k]; //change in error with respect to bias is just error/activation, since activation/bias is 1 (bias is constant)
                    }

                    errorPerActivation[i][j] *= activation; //multiply error in activation by derivative of compression to complete the gradient

                    for (int k = 0; k < hiddenLayers[i].weights.ColumnCount; k++) //adjust every weight
                    {
                        steps[i].weightStep[j, k] = errorPerActivation[i][j] * previousLayer.neurons[k];
                    }
                }
            }

            return steps;
        }

        private void applyStep(GradStep step, DerivedLayer layer) 
        {
            for(int i = 0; i < layer.weights.RowCount; i++) //iterate first by column because every column represents a neuron
            {
                for(int j = 0; j < layer.weights.ColumnCount; j++) //each row represents the jth weight for each neuron
                {
                    layer.weights[i, j] -= learningFactor * step.weightStep[i, j]; //change each weight by its negative gradient
                }

                layer.biases[i] -= learningFactor * step.biasStep[i]; //change the bias by its negative gradient, one for each neuron in the layer
            }
        }

        public int applySteps(GradStep[] steps, double[] expected) //applies a set of gradient steps returned by backpropogate()
        {

            for(int i = 0; i < steps.Count() - 1; i++) // FACTOR is an unused property that could be optimized for, any implementation i had made training slower
            {
                applyStep(steps[i], hiddenLayers[i]); //exploits interference between waveforms of multiple functions to apply the applyStep() method to each step
            }
            applyStep(steps[steps.Count() - 1], output); //then does it to the output layer, since output is not included in the hiddenLayers array

            try
            {
                check(expected); //change in weights should be proportional to the total error over number of weights+biases
            }
            catch (IndexOutOfRangeException)
            {
                return -1; //kill this network
            }
            catch (DataMisalignedException)
            {
                return -2; //kill this network with fire
            }
            return 0; //keep training this, all is well
        }
    }
}