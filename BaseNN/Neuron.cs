using System.Numerics;

namespace BaseNN
{
    public class Neuron
    {
        private const byte WEIGHTSCOUNT = 2;
        private float _bias;
        private Vector<float> _weights;

        public Neuron(float bias, Vector<float> weights)
        {
            _bias = bias;
            _weights = weights;
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

        public byte Feed(Vector<float> inputs)
        {
            float z = 0;
            for (int i = 0; i < WEIGHTSCOUNT; i++)
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
