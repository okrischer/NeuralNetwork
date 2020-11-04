using System.Numerics;

namespace BaseNN
{
    public class Neuron
    {
        private int _weightsCount;
        private float _bias;
        private Vector<float> _weights;

        public Neuron(float bias, Vector<float> weights)
        {
            _bias = bias;
            _weights = weights;
            _weightsCount = Vector<float>.Count;
        }

        public Neuron(float bias, Vector<float> weights, int weightsCount)
        {
            _bias = bias;
            _weights = weights;
            _weightsCount = weightsCount;
        }

        public float Bias
        {
            get => _bias; 
            set => _bias = value;
        }
        public Vector<float> Weights
        {
            get => _weights;
            set => _weights = value;
        }
        public int WeightsCount
        {
            get => _weightsCount;
            set => _weightsCount = value;
        }

        public byte Feed(Vector<float> inputs)
        {
            float z = 0;
            for (int i = 0; i < _weightsCount; i++)
            {
                z += inputs[i] * _weights[i];
            }
            z += _bias;

            if (z > 0)
            {
                return (byte) 1;
            }
            else 
            {
                return (byte) 0;
            }
        }
    }
}
