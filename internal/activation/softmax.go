package activation

import "math"

// Softmax activation function.
func Softmax(z, sum float64) float64 {
	return math.Exp(z) / sum
}
