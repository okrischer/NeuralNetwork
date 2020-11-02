using System;

namespace BaseNN
{
    public class Neuron
    {
        private float _bias;
        private float[] _weights;

        public Neuron(float bias, float[] weights)
        {
            _bias = bias;
            _weights = weights;
        }

        public float Bias
        {
            get => _bias; 
            set => _bias = value;
        }
        public float[] Weights
        {
            get => _weights;
            set => _weights = value;
        }

        public byte Feed(float[] input)
        {
            if (input.Length != _weights.Length)
            {
                throw new OperationCanceledException(
                    "length of input doesn`t match length of weights");
            }
            float z =0;
            for (int i = 0; i < input.Length; i++)
            {
                z += input[i] * _weights[i];
            }
            z += _bias;
            
            if (z > 0) return (byte) 1;
            else return (byte) 0;
        }
    }
}
