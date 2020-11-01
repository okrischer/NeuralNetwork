using System;

namespace BaseNN
{
    public class Neuron
    {
        private double _bias;
        private double[] _weights;

        public Neuron(double bias, double[] weights)
        {
            _bias = bias;
            _weights = weights;
        }

        public double Bias
        {
            get => _bias; 
            set => _bias = value;
        }
        public double[] Weights
        {
            get => _weights;
            set => _weights = value;
        }

        public byte Feed(double[] input)
        {
            double z = 0;
            if (input.Length != _weights.Length)
            {
                throw new OperationCanceledException(
                    "the number of input values does not match the number of weights");
            } 
            else
            {
                for (int i = 0; i < input.Length; i++)
                {
                    z += input[i] * _weights[i];
                }
            }
            z += _bias;
            
            if (z > 0) return (byte) 1;
            else return (byte) 0;
        }
    }
}
