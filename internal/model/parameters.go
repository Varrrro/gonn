package model

// Parameters for neural network training.
type Parameters struct {
	initialEta float64
	Eta        float64
}

// CreateParameters for neural network training.
func CreateParameters(eta float64) Parameters {
	return Parameters{
		initialEta: eta,
		Eta:        eta,
	}
}

// UpdateEta for the given epoch.
func (p *Parameters) UpdateEta(epoch int) {
	p.Eta = p.initialEta / float64(1+epoch/2)
}
