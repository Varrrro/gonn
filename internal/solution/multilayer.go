package solution

import (
	"github.com/varrrro/gonn/internal/layer"
	"github.com/varrrro/gonn/internal/model"
	"gonum.org/v1/gonum/mat"
)

// InitMultilayer network with the given training and test sets.
func InitMultilayer(trainImgs, testImgs *[]mat.Vector, trainLabels, testLabels *[]int) {
	hiddenLayer := layer.CreateSigmoidalLayer(784, 256)
	outputLayer := layer.CreateSoftmaxLayer(256, 10)

	params := model.CreateDefaultParameters()

	nn := model.Network{Layers: []model.Layer{hiddenLayer, outputLayer}, Params: params}

	nn.Train(*trainImgs, *trainLabels, 10)

	//nn.Test(*trainImgs, *trainLabels)
	nn.Test(*testImgs, *testLabels)
}
