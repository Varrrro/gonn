package layer

import (
	"math"

	"github.com/varrrro/gonn/internal/activation"
	"github.com/varrrro/gonn/internal/util"
	"gonum.org/v1/gonum/mat"
)

// SoftmaxLayer wich outputs amount to 1.
type SoftmaxLayer struct {
	InputSize    int
	OutputSize   int
	Input        mat.Vector
	Output       mat.Vector
	Weights      mat.Matrix
	WeightDeltas mat.Matrix
	NeuronDeltas mat.Vector
}

// CreateSoftmaxLayer with the given values.
func CreateSoftmaxLayer(nInput, nOutput int, weights *[]float64) *SoftmaxLayer {
	return &SoftmaxLayer{
		InputSize:    nInput,
		OutputSize:   nOutput,
		Weights:      mat.NewDense(nOutput, nInput, *weights),
		WeightDeltas: mat.NewDense(nOutput, nInput, util.InitializeWeightDeltas(nInput, nOutput)),
	}
}

// GetOutput of this layer.
func (l *SoftmaxLayer) GetOutput() mat.Vector {
	return l.Output
}

// GetNeuronDeltas of this layer.
func (l *SoftmaxLayer) GetNeuronDeltas() mat.Vector {
	return l.NeuronDeltas
}

// GetWeights of this layer.
func (l *SoftmaxLayer) GetWeights() mat.Matrix {
	return l.Weights
}

// FeedForward a set of features through this layer.
func (l *SoftmaxLayer) FeedForward(features mat.Vector) {
	l.Input = features

	output := mat.NewVecDense(l.OutputSize, nil)
	z := mat.NewVecDense(l.OutputSize, nil)
	z.MulVec(l.Weights, features)

	sum := 0.0
	for i := 0; i < l.OutputSize; i++ {
		sum += math.Exp(z.AtVec(i))
	}

	for i := 0; i < l.OutputSize; i++ {
		output.SetVec(i, activation.Softmax(z.AtVec(i), sum))
	}

	l.Output = output
}

// CalculateNeuronDeltas with the error gradient.
func (l *SoftmaxLayer) CalculateNeuronDeltas(gradient mat.Vector) {
	deltas := mat.NewVecDense(l.OutputSize, nil)

	for i := 0; i < l.OutputSize; i++ {
		value := l.Output.AtVec(i) * (1 - l.Output.AtVec(i)) * gradient.AtVec(i)
		deltas.SetVec(i, value)
	}

	l.NeuronDeltas = deltas
}

// CalculateGradient of the error in this layer based on the next.
func (l *SoftmaxLayer) CalculateGradient(deltas mat.Vector, weights mat.Matrix) mat.Vector {
	gradient := mat.NewVecDense(l.OutputSize, nil)
	n, _ := weights.Dims()

	for i := 0; i < l.OutputSize; i++ {
		sum := 0.0

		for j := 0; j < n; j++ {
			sum += weights.At(j, i) * deltas.AtVec(j)
		}

		gradient.SetVec(i, sum)
	}

	return gradient
}

// DoMomentumStep with the given Mu.
func (l *SoftmaxLayer) DoMomentumStep(mu float64) {
	increment := mat.NewDense(l.OutputSize, l.InputSize, nil)
	newWeights := mat.NewDense(l.OutputSize, l.InputSize, nil)

	increment.Scale(mu, l.WeightDeltas)
	newWeights.Add(l.Weights, increment)

	l.Weights = newWeights
}

// DoCorrectionStep with the given Eta.
func (l *SoftmaxLayer) DoCorrectionStep(eta float64) {
	deltas := mat.NewDense(l.OutputSize, l.InputSize, nil)
	newWeights := mat.NewDense(l.OutputSize, l.InputSize, nil)

	for i := 0; i < l.OutputSize; i++ {
		for j := 0; j < l.InputSize; j++ {
			value := l.Input.AtVec(j) * l.NeuronDeltas.AtVec(i) * eta * -1
			deltas.Set(i, j, value)
		}
	}

	newWeights.Add(l.Weights, deltas)

	l.Weights = newWeights
	l.WeightDeltas = deltas
}
