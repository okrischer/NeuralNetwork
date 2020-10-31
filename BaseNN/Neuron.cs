using System;

namespace BaseNN
{
    public class Neuron
    {
        private double bias;
        private double[] weights;

        public Neuron(double bias, double[] weights)
        {
            this.bias = bias;
            this.weights = weights;
        }

        public double Bias
        {
            get
            {
                return bias;
            }
            set
            {
                bias = value;
            }
        }
        public double[] Weights
        {
            get
            {
                return weights;
            }
            set
            {
                weights = value;
            }
        }
        public byte Feed(double[] input)
        {
            double z = 0;
            if (input.Length != weights.Length)
            {
                throw new OperationCanceledException(
                    "the number of input values does not match the number of weights");
            } 
            else
            {
                for (int i = 0; i < input.Length; i++)
                {
                    z += input[i] * weights[i];
                }
            }
            z += bias;
            if (z > 0)
            {
                return (byte) 1;
            }
            else
            {
                return (byte) 1;
            }
        }
    }
}
