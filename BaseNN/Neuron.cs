using System.Numerics;

namespace BaseNN
{
    public class Neuron
    {
        private Vector<float> _weights;

        public Neuron(float bias, Vector<float> weights,
                int weightsCount = 8, float learningRate = 0.1F)
        {
            Bias = bias;
            _weights = weights;
            WeightsCount = weightsCount;
            LearningRate = learningRate;
        }

        public float Bias { get; set; }
        public float LearningRate { get; set; }
        public int WeightsCount { get; set; }

        public Vector<float> Weights
        {
            get => _weights;
            set => _weights = value;
        }

        public byte Feed(Vector<float> inputs)
        {
            float z = 0;
            for (int i = 0; i < WeightsCount; i++)
            {
                z += inputs[i] * _weights[i];
            }
            z += Bias;

            return z > 0 ? 1 : 0;
        }

        public void Train(Vector<float> inputs, int target)
        {
            var error = target - Feed(inputs);
            _weights += inputs * (LearningRate * error);
            Bias += LearningRate * error;
        }
    }
}
