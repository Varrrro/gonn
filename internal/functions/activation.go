package functions

import "math"

// Logistic activation function.
func Logistic(z float64) float64 {
	return 1 / (1 + math.Exp(-1.0*z))
}

// Softmax activation function.
func Softmax(c, z, sum float64) float64 {
	return math.Exp(z-c) / sum
}

// Relu activation function.
func Relu(z float64) float64 {
	return math.Max(0.0, z)
}
