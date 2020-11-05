using System.Numerics;

namespace BaseNN
{
    public class Neuron
    {
        private int _weightsCount;
        private float _bias;
        private Vector<float> _weights;
        private float _learningRate;

        public Neuron(float bias, Vector<float> weights,
                int weightsCount = 8, float learningRate = 0.1F)
        {
            _bias = bias;
            _weights = weights;
            _weightsCount = weightsCount;
            _learningRate = learningRate;
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
        public float LearningRate
        {
            get => _learningRate;
            set => _learningRate = value;
        }

        public byte Feed(Vector<float> inputs)
        {
            float z = 0;
            for (int i = 0; i < _weightsCount; i++)
            {
                z += inputs[i] * _weights[i];
            }
            z += _bias;

            if (z > 0) return 1;
            else return 0;
        }

        public void Train(Vector<float> inputs, int target)
        {
            int output = Feed(inputs);
            int error = target - output;
            float adjustment = _learningRate * error;
            _weights += inputs * adjustment;
            _bias += adjustment;
        }
    }
}
