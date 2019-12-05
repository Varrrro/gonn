package solution

import (
	"github.com/varrrro/gonn/internal/layer"
	"github.com/varrrro/gonn/internal/model"
	"gonum.org/v1/gonum/mat"

	"github.com/varrrro/gonn/internal/util"
)

// InitMultilayer network with the given training and test sets.
func InitMultilayer(trainImgs, testImgs *[]mat.Vector, trainLabels, testLabels *[]int) {
	// Initialize weight and bias arrays
	hiddenWeights := util.InitializeRandom(784 * 256)
	hiddenBiases := util.InitializeRandom(256)
	outputWeights := util.InitializeRandom(256 * 10)
	outputBiases := util.InitializeRandom(10)

	// Create the layers
	hiddenLayer := layer.CreateSigmoidalLayer(784, 256, hiddenWeights, hiddenBiases)
	outputLayer := layer.CreateSoftmaxLayer(256, 10, outputWeights, outputBiases)

	params := model.CreateDefaultParameters()

	// Create the network
	nn := model.Network{Layers: []model.Layer{hiddenLayer, outputLayer}, Params: params}

	// Train the network with the training set for a number of epochs
	nn.Train(*trainImgs, *trainLabels, 10)

	// Test the network with the test set
	nn.Test(*testImgs, *testLabels)
}
