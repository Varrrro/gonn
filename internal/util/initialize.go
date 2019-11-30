package util

import "math/rand"

// InitializeWeights array with values [-1, 1].
func InitializeWeights(nInput, nOutput int) []float64 {
	weights := make([]float64, nInput*nOutput)

	for i := 0; i < len(weights); i++ {
		weights[i] = (rand.Float64() * 2) - 1
	}

	return weights
}

// InitializeWeightDeltas with zero values.
func InitializeWeightDeltas(nInput, nOutput int) []float64 {
	deltas := make([]float64, nInput*nOutput)

	for i := 0; i < len(deltas); i++ {
		deltas[i] = 0.0
	}

	return deltas
}
