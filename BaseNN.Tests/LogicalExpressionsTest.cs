using System;
using Xunit;
using BaseNN;
using System.Numerics;

namespace BaseNN.Tests
{
    public class LogicalAndTest
    {
        // bias and weights for logical `AND` tests
        static float Bias = -1.5F;
        static float[] weights = {1,1,0,0,0,0,0,0};
        static Vector<float> Weights = new Vector<float>(weights);
        Neuron p = new Neuron(Bias,  Weights, 2);

        [Theory]
        [InlineData(1,1)]
        public void TestLogicalAndYieldsTrue(float v0, float v1)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            inputs[1] = v1;
            Assert.Equal(1, p.Feed(new Vector<float>(inputs)));
        }

        [Theory]
        [InlineData(1,0)]
        [InlineData(0,1)]
        [InlineData(0,0)]
        public void TestLogicalAndYieldsFalse(float v0, float v1)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            inputs[1] = v1;
            Assert.Equal(0, p.Feed(new Vector<float>(inputs)));
        }
    }

    public class LogicalOrTest
    {
        // bias and weights for logical `OR` tests
        static float Bias = -0.5F;
        static float[] weights = {1,1,0,0,0,0,0,0};
        static Vector<float> Weights = new Vector<float>(weights);
        Neuron p = new Neuron(Bias,  Weights, 2);

        [Theory]
        [InlineData(1,1)]
        [InlineData(1,0)]
        [InlineData(0,1)]
        public void TestLogicalOrYieldsTrue(float v0, float v1)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            inputs[1] = v1;
            Assert.Equal(1, p.Feed(new Vector<float>(inputs)));
        }

        [Theory]
        [InlineData(0,0)]
        public void TestLogicalOrYieldsFalse(float v0, float v1)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            inputs[1] = v1;
            Assert.Equal(0, p.Feed(new Vector<float>(inputs)));
        }
    }

    public class LogicalNotTest
    {
        // bias and weights for logical `NOT` tests
        static float Bias = 0.5F;
        static float[] weights = {-1,0,0,0,0,0,0,0};
        static Vector<float> Weights = new Vector<float>(weights);
        Neuron p = new Neuron(Bias,  Weights, 1);

        [Theory]
        [InlineData(0)]
        public void TestLogicalNotYieldsTrue(float v0)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            Assert.Equal(1, p.Feed(new Vector<float>(inputs)));
        }

        [Theory]
        [InlineData(1)]
        public void TestLogicalNotYieldsFalse(float v0)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            Assert.Equal(0, p.Feed(new Vector<float>(inputs)));
        }
    }
}
