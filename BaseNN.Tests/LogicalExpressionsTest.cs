using System;
using Xunit;
using BaseNN;
using System.Collections.Generic;

namespace BaseNN.Tests
{
    public static class TestData
    {
        // create all 4 permutations for true/false values
        private static float[] p1q1 = {1,1};
        private static float[] p1q0 = {1,0};
        private static float[] p0q1 = {0,1};
        private static float[] p0q0 = {0,0};
        private static List<float[]> inputs = new List<float[]>();
        public static List<float[]> GetInputs()
        {
            inputs.Add(p1q1);
            inputs.Add(p1q0);
            inputs.Add(p0q1);
            inputs.Add(p0q0);

            return inputs;
        }
    }

    public class LogicalAndTest
    {
        // bias and weights for logical `AND` tests
        static float bias = -1.5F;
        static float[] weights = {1,1};
        Neuron p = new Neuron(bias,  weights);


        [Fact]
        public void TestLogicalAndTrue()
        {
            float[] input = TestData.GetInputs()[0];
            Assert.Equal(1, p.Feed(input));
        }

        [Fact]
        public void TestLogicalAndFalse()
        {
            var inputs = TestData.GetInputs();
            for (int i = 1; i < 4; i++)
            {
                Assert.Equal(0, p.Feed(inputs[i]));
            }
        }
    }
}
