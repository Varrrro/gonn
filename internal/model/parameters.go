package model

const defaultEta = 0.1
const defaultMu = 0.5

// Parameters for neural network training.
type Parameters struct {
	Eta float64
	Mu  float64
}

// CreateDefaultParameters for neural network training.
func CreateDefaultParameters() Parameters {
	return Parameters{
		Eta: defaultEta,
		Mu:  defaultMu,
	}
}

// SetEta to the given value.
func (p *Parameters) SetEta(v float64) {
	p.Eta = v
}

// SetMu to the given value.
func (p *Parameters) SetMu(v float64) {
	p.Mu = v
}
