using System;
using Xunit;
using BaseNN;
using System.Numerics;

namespace BaseNN.Tests
{
    public class DigitalComparatorTest
    {
        // create vectors from floats
        private static Vector<float> CreateVector(float p, float q = 0)
        {
            float[] weights = new float[8];
            weights[0] = p;
            weights[1] = q;
            return new Vector<float>(weights);
        }
        
        // create the needed perceptrons
        Neuron And = new Neuron(-1.5F,  CreateVector(1,1), 2);
        Neuron Nor = new Neuron(0.5F,  CreateVector(-1,-1), 2);
        Neuron Not = new Neuron(0.5F,  CreateVector(-1), 1);

        [Theory]
        [InlineData(0,1)]
        public void DigitalComparatorYieldsAlowerB(float a, float b)
        {
            float notA = Not.Feed(CreateVector(a));
            float AlowerB = And.Feed(CreateVector(notA, b));

            Assert.Equal(1, AlowerB);
        }

        [Theory]
        [InlineData(1,0)]
        public void DigitalComparatorYieldsAgreaterB(float a, float b)
        {
            float notB = Not.Feed(CreateVector(b));
            float AgreaterB = And.Feed(CreateVector(a, notB));

            Assert.Equal(1, AgreaterB);
        }

        [Theory]
        [InlineData(1,1)]
        [InlineData(0,0)]
        public void DigitalComparatorYieldsAEqualB(float a, float b)
        {
            float notA = Not.Feed(CreateVector(a));
            float notB = Not.Feed(CreateVector(b));

            float AlowerB = And.Feed(CreateVector(notA, b));
            float AgreaterB = And.Feed(CreateVector(a, notB));
            float AequalB = Nor.Feed(CreateVector(AlowerB, AgreaterB));

            Assert.Equal(1, AequalB);
        }
    }
}
